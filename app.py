"""
=============================================================================
  AI-Powered Nutrition Agent — Flask + IBM Watsonx.ai (Granite)
=============================================================================
  Author   : IBM Bob
  Stack    : Python 3.10+, Flask, ibm-watsonx-ai, python-dotenv
  Model    : ibm/granite-3-3-8b-instruct  (configurable below)
=============================================================================

  ╔══════════════════════════════════════════════════════════════════════╗
  ║                     AGENT_INSTRUCTIONS                               ║
  ║  Edit the constants in this section to customise the agent without   ║
  ║  touching any other part of the code.                                ║
  ╚══════════════════════════════════════════════════════════════════════╝
"""

# ── AGENT INSTRUCTIONS ────────────────────────────────────────────────────────

# 1. PERSONA & TONE
AGENT_NAME = "NutriBot"
AGENT_TONE = "friendly, empathetic, and professional"
AGENT_TAGLINE = "Your personalised AI nutrition companion"

# 2. DIET SPECIALISATION
#    Options: "balanced", "vegetarian", "vegan", "keto", "diabetic", "ayurvedic"
DEFAULT_DIET_MODE = "balanced"

# 3. CULTURAL / REGIONAL FOOD PREFERENCE
#    Set to True to bias meal suggestions toward Indian cuisine
INDIAN_FOOD_PREFERENCE = True
INDIAN_FOOD_NOTE = (
    "Prefer Indian regional cuisine wherever possible — include dishes like "
    "dal, sabzi, roti, rice, idli, dosa, upma, khichdi, poha, and traditional "
    "breakfast / lunch / dinner plates.  Use Indian portion references "
    "(e.g. katori, chapati, glass) and spices (turmeric, cumin, coriander)."
)

# 4. SAFETY RULES (appended verbatim to every system prompt)
SAFETY_RULES = """
SAFETY RULES (non-negotiable):
- Never prescribe medicines or medical treatments.
- Always advise consulting a registered dietitian or doctor for medical conditions.
- Do not provide advice that could lead to extreme calorie restriction (< 1 200 kcal/day for adults).
- Flag any query about eating disorders with a compassionate referral to professional help.
- Do not make absolute disease-cure claims; use hedged language ("may help", "evidence suggests").
"""

# 5. RESPONSE FORMAT PREFERENCES
RESPONSE_LANGUAGE = "English"          # e.g. "Hindi", "Tamil", "Bengali"
USE_EMOJIS_IN_RESPONSES = True         # set False for clinical / plain-text mode
MAX_RESPONSE_TOKENS = 1200

# 6. WATSONX MODEL CONFIGURATION
WATSONX_MODEL_ID = "meta-llama/llama-3-3-70b-instruct"
WATSONX_TEMPERATURE = 0.7
WATSONX_TOP_P = 0.9
WATSONX_TOP_K = 50
WATSONX_MAX_NEW_TOKENS = MAX_RESPONSE_TOKENS

# ── END AGENT INSTRUCTIONS ────────────────────────────────────────────────────

import os
import json
import logging
from datetime import datetime
from functools import lru_cache
from pathlib import Path

from flask import Flask, request, jsonify, render_template, session
from flask_cors import CORS
from dotenv import load_dotenv

# IBM Watsonx.ai SDK
from ibm_watsonx_ai import APIClient, Credentials
from ibm_watsonx_ai.foundation_models import ModelInference
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams

# ── Bootstrap ─────────────────────────────────────────────────────────────────

# Always load .env from the same directory as app.py,
# regardless of which working directory the server is launched from.
# Falls back to .env.example if .env is missing.
_BASE = Path(__file__).parent
_ENV_PATH = _BASE / ".env"
_ENV_EXAMPLE_PATH = _BASE / ".env.example"

if _ENV_PATH.exists():
    load_dotenv(dotenv_path=_ENV_PATH, override=True)
    print(f"[env] Loaded: {_ENV_PATH}")
elif _ENV_EXAMPLE_PATH.exists():
    # override=False so real environment variables (e.g. from Render) take priority
    load_dotenv(dotenv_path=_ENV_EXAMPLE_PATH, override=False)
    print(f"[env] WARNING: .env not found — loaded from .env.example instead (env vars take priority)")
else:
    print("[env] No .env file found — using environment variables only")

# Startup credential check — fail fast with a clear message
_missing = [v for v in ("IBM_CLOUD_API_KEY", "WATSONX_PROJECT_ID") if not os.getenv(v)]
if _missing:
    raise EnvironmentError(
        f"\n\n  *** STARTUP FAILED ***\n"
        f"  Missing environment variables: {', '.join(_missing)}\n"
        f"  Edit  {_ENV_PATH}  and set the missing values.\n"
        f"  Current .env contents:\n"
        + "\n".join(
            f"    {line}" for line in (_ENV_PATH.read_text() if _ENV_PATH.exists() else "(file missing)").splitlines()
        )
    )

print(f"[env] IBM_CLOUD_API_KEY  = {os.getenv('IBM_CLOUD_API_KEY','')[:8]}***")
print(f"[env] WATSONX_PROJECT_ID = {os.getenv('WATSONX_PROJECT_ID','')[:8]}***")
print(f"[env] WATSONX_URL        = {os.getenv('WATSONX_URL','')}")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
log = logging.getLogger(__name__)

app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "change-me-in-production")
CORS(app)

# ── Watsonx client (lazy-initialised once) ────────────────────────────────────

_watsonx_model: ModelInference | None = None


def get_watsonx_model() -> ModelInference:
    """Return a cached Watsonx ModelInference instance."""
    global _watsonx_model
    if _watsonx_model is not None:
        return _watsonx_model

    api_key    = (os.getenv("IBM_CLOUD_API_KEY") or "").strip()
    project_id = (os.getenv("WATSONX_PROJECT_ID") or "").strip()
    url        = (os.getenv("WATSONX_URL") or "https://us-south.ml.cloud.ibm.com").strip()

    log.info("Credentials check — api_key present: %s, project_id present: %s, url: %s",
             bool(api_key), bool(project_id), url)

    if not api_key:
        raise EnvironmentError(
            "IBM_CLOUD_API_KEY is missing or empty in .env — "
            "open nutrition_agent/.env and set IBM_CLOUD_API_KEY=<your key>"
        )
    if not project_id:
        raise EnvironmentError(
            "WATSONX_PROJECT_ID is missing or empty in .env — "
            "open nutrition_agent/.env and set WATSONX_PROJECT_ID=<your project id>"
        )

    credentials = Credentials(url=url, api_key=api_key)
    client      = APIClient(credentials=credentials, project_id=project_id)

    _watsonx_model = ModelInference(
        model_id   = WATSONX_MODEL_ID,
        api_client = client,
        params={
            GenParams.DECODING_METHOD : "sample",
            GenParams.TEMPERATURE     : WATSONX_TEMPERATURE,
            GenParams.TOP_P           : WATSONX_TOP_P,
            GenParams.TOP_K           : WATSONX_TOP_K,
            GenParams.MAX_NEW_TOKENS  : WATSONX_MAX_NEW_TOKENS,
            GenParams.STOP_SEQUENCES  : ["Human:", "User:"],
        },
    )
    log.info("Watsonx ModelInference initialised — model: %s", WATSONX_MODEL_ID)
    return _watsonx_model


# ── Prompt builders ───────────────────────────────────────────────────────────

def _build_system_prompt(context: dict) -> str:
    """Compose the system prompt from AGENT_INSTRUCTIONS + user context."""
    emoji_note = (
        "Use relevant food/health emojis to make responses engaging."
        if USE_EMOJIS_IN_RESPONSES
        else "Do NOT use emojis; keep responses plain text."
    )
    indian_note = INDIAN_FOOD_NOTE if INDIAN_FOOD_PREFERENCE else ""
    diet_note   = f"Default diet mode: {DEFAULT_DIET_MODE}."

    user_ctx = ""
    if context:
        parts = []
        if context.get("name"):
            parts.append(f"User name: {context['name']}")
        if context.get("age"):
            parts.append(f"Age: {context['age']}")
        if context.get("weight"):
            parts.append(f"Weight: {context['weight']} kg")
        if context.get("height"):
            parts.append(f"Height: {context['height']} cm")
        if context.get("goal"):
            parts.append(f"Goal: {context['goal']}")
        if context.get("conditions"):
            parts.append(f"Health conditions: {context['conditions']}")
        if context.get("allergies"):
            parts.append(f"Allergies/intolerances: {context['allergies']}")
        if parts:
            user_ctx = "USER PROFILE:\n" + "\n".join(f"  • {p}" for p in parts)

    return f"""You are {AGENT_NAME}, {AGENT_TAGLINE}.
Tone: {AGENT_TONE}.
Language: {RESPONSE_LANGUAGE}.
{emoji_note}
{diet_note}
{indian_note}

Your capabilities:
  • Personalised nutrition plans and calorie analysis
  • Healthy meal suggestions (breakfast, lunch, dinner, snacks)
  • BMI interpretation and healthy-weight guidance
  • Family / group diet recommendations
  • Ingredient substitution and recipe tips
  • Hydration, micronutrient, and supplement guidance

{user_ctx}

{SAFETY_RULES}
Always respond in a structured, easy-to-read format with clear headings.
"""


def _chat_prompt(system: str, history: list[dict], user_msg: str) -> str:
    """Assemble a full prompt string for the Granite chat format."""
    lines = [f"<|system|>\n{system.strip()}\n<|user|>"]
    for turn in history[-6:]:          # keep last 6 turns for context window
        lines.append(turn["content"])
        if turn["role"] == "user":
            lines.append("<|assistant|>")
        else:
            lines.append("<|user|>")
    lines.append(user_msg)
    lines.append("<|assistant|>")
    return "\n".join(lines)


# ── Nutrition helpers ─────────────────────────────────────────────────────────

def calculate_bmi(weight_kg: float, height_cm: float) -> dict:
    h = height_cm / 100
    bmi = round(weight_kg / (h * h), 1)
    if bmi < 18.5:
        category, advice = "Underweight", "Focus on nutrient-dense, calorie-rich foods."
    elif bmi < 25:
        category, advice = "Normal weight", "Maintain your current healthy habits."
    elif bmi < 30:
        category, advice = "Overweight", "Aim for a moderate calorie deficit and regular activity."
    else:
        category, advice = "Obese", "Consult a doctor or dietitian for a medically supervised plan."
    return {"bmi": bmi, "category": category, "advice": advice}


def calculate_tdee(weight_kg: float, height_cm: float,
                   age: int, gender: str, activity: str) -> dict:
    """Harris-Benedict BMR × activity multiplier."""
    if gender.lower() == "female":
        bmr = 447.593 + 9.247 * weight_kg + 3.098 * height_cm - 4.330 * age
    else:
        bmr = 88.362 + 13.397 * weight_kg + 4.799 * height_cm - 5.677 * age

    multipliers = {
        "sedentary"   : 1.2,
        "light"       : 1.375,
        "moderate"    : 1.55,
        "active"      : 1.725,
        "very_active" : 1.9,
    }
    factor = multipliers.get(activity.lower(), 1.55)
    tdee   = round(bmr * factor)
    return {
        "bmr"          : round(bmr),
        "tdee"         : tdee,
        "weight_loss"  : tdee - 500,
        "weight_gain"  : tdee + 300,
    }


# ── API Routes ────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html", agent_name=AGENT_NAME, tagline=AGENT_TAGLINE)


@app.route("/api/chat", methods=["POST"])
def chat():
    """Main chat endpoint — calls Watsonx Granite."""
    data       = request.get_json(force=True)
    user_msg   = (data.get("message") or "").strip()
    history    = data.get("history", [])
    user_ctx   = data.get("context", {})

    if not user_msg:
        return jsonify({"error": "Empty message"}), 400

    try:
        model  = get_watsonx_model()
        system = _build_system_prompt(user_ctx)
        prompt = _chat_prompt(system, history, user_msg)

        result  = model.generate_text(prompt=prompt)
        reply   = result.strip() if isinstance(result, str) else result

        return jsonify({
            "reply"    : reply,
            "timestamp": datetime.utcnow().isoformat(),
            "model"    : WATSONX_MODEL_ID,
        })

    except EnvironmentError as e:
        log.error("Config error: %s", e)
        return jsonify({"error": str(e)}), 503
    except Exception as e:
        log.exception("Watsonx call failed")
        return jsonify({"error": f"AI service error: {str(e)}"}), 500


@app.route("/api/nutrition-plan", methods=["POST"])
def nutrition_plan():
    """Generate a structured weekly nutrition plan."""
    data    = request.get_json(force=True)
    profile = data.get("profile", {})

    prompt_user = (
        f"Create a detailed 7-day nutrition plan for:\n"
        f"Name: {profile.get('name','User')}, "
        f"Age: {profile.get('age','30')}, "
        f"Weight: {profile.get('weight','70')} kg, "
        f"Height: {profile.get('height','170')} cm, "
        f"Goal: {profile.get('goal','maintain weight')}, "
        f"Diet: {profile.get('diet', DEFAULT_DIET_MODE)}, "
        f"Conditions: {profile.get('conditions','none')}, "
        f"Allergies: {profile.get('allergies','none')}.\n\n"
        f"Include breakfast, lunch, dinner, and two snacks per day. "
        f"Add estimated calories and key nutrients for each day. "
        f"Conclude with a weekly summary and top-3 tips."
    )

    try:
        model  = get_watsonx_model()
        system = _build_system_prompt(profile)
        prompt = _chat_prompt(system, [], prompt_user)
        result = model.generate_text(prompt=prompt)
        return jsonify({"plan": result.strip(), "profile": profile})
    except Exception as e:
        log.exception("nutrition-plan failed")
        return jsonify({"error": str(e)}), 500


@app.route("/api/calorie-analysis", methods=["POST"])
def calorie_analysis():
    """Analyse a meal or food list for calories and nutrients."""
    data  = request.get_json(force=True)
    foods = data.get("foods", "")

    prompt_user = (
        f"Analyse the following foods/meal and provide:\n"
        f"1. Estimated calories per item\n"
        f"2. Total calories\n"
        f"3. Macronutrient breakdown (protein, carbs, fat, fibre)\n"
        f"4. Key micronutrients present\n"
        f"5. Health score (1-10) with reasoning\n"
        f"6. Two healthier swap suggestions\n\n"
        f"Foods: {foods}"
    )

    try:
        model  = get_watsonx_model()
        system = _build_system_prompt({})
        prompt = _chat_prompt(system, [], prompt_user)
        result = model.generate_text(prompt=prompt)
        return jsonify({"analysis": result.strip(), "foods": foods})
    except Exception as e:
        log.exception("calorie-analysis failed")
        return jsonify({"error": str(e)}), 500


@app.route("/api/family-plan", methods=["POST"])
def family_plan():
    """Generate diet recommendations for multiple family members."""
    data    = request.get_json(force=True)
    members = data.get("members", [])

    members_text = "\n".join(
        f"  • {m.get('name','Member')} | Age {m.get('age','?')} | "
        f"Goal: {m.get('goal','healthy eating')} | "
        f"Conditions: {m.get('conditions','none')}"
        for m in members
    )

    prompt_user = (
        f"Create a unified family nutrition plan for the following members:\n"
        f"{members_text}\n\n"
        f"Provide:\n"
        f"1. A common family meal plan for 3 days (breakfast, lunch, dinner)\n"
        f"2. Individual modifications for each member\n"
        f"3. Shopping list for the 3 days\n"
        f"4. Family-friendly healthy snack ideas"
    )

    try:
        model  = get_watsonx_model()
        system = _build_system_prompt({})
        prompt = _chat_prompt(system, [], prompt_user)
        result = model.generate_text(prompt=prompt)
        return jsonify({"plan": result.strip(), "members": members})
    except Exception as e:
        log.exception("family-plan failed")
        return jsonify({"error": str(e)}), 500


@app.route("/api/bmi", methods=["POST"])
def bmi_api():
    data = request.get_json(force=True)
    try:
        weight = float(data["weight"])
        height = float(data["height"])
        result = calculate_bmi(weight, height)
        return jsonify(result)
    except (KeyError, ValueError) as e:
        return jsonify({"error": f"Invalid input: {e}"}), 400


@app.route("/api/tdee", methods=["POST"])
def tdee_api():
    data = request.get_json(force=True)
    try:
        result = calculate_tdee(
            weight_kg = float(data["weight"]),
            height_cm = float(data["height"]),
            age       = int(data["age"]),
            gender    = data.get("gender", "male"),
            activity  = data.get("activity", "moderate"),
        )
        return jsonify(result)
    except (KeyError, ValueError) as e:
        return jsonify({"error": f"Invalid input: {e}"}), 400


@app.route("/api/meal-suggestions", methods=["POST"])
def meal_suggestions():
    """Quick meal ideas based on available ingredients or preferences."""
    data     = request.get_json(force=True)
    meal_type = data.get("meal_type", "lunch")
    prefs    = data.get("preferences", "")
    ingredients = data.get("ingredients", "")

    prompt_user = (
        f"Suggest 5 {meal_type} options"
        + (f" using: {ingredients}" if ingredients else "")
        + (f". Preferences: {prefs}" if prefs else "")
        + ". For each: name, ingredients, quick prep steps, calories, and key nutrients."
    )

    try:
        model  = get_watsonx_model()
        system = _build_system_prompt({})
        prompt = _chat_prompt(system, [], prompt_user)
        result = model.generate_text(prompt=prompt)
        return jsonify({"suggestions": result.strip()})
    except Exception as e:
        log.exception("meal-suggestions failed")
        return jsonify({"error": str(e)}), 500


@app.route("/api/health")
def health_check():
    return jsonify({
        "status"     : "ok",
        "agent"      : AGENT_NAME,
        "model"      : WATSONX_MODEL_ID,
        "diet_mode"  : DEFAULT_DIET_MODE,
        "indian_food": INDIAN_FOOD_PREFERENCE,
        "timestamp"  : datetime.utcnow().isoformat(),
    })


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    port  = int(os.getenv("PORT", 5000))
    debug = os.getenv("FLASK_DEBUG", "false").lower() == "true"
    log.info("Starting %s on port %d  (debug=%s)", AGENT_NAME, port, debug)
    app.run(host="0.0.0.0", port=port, debug=debug)
