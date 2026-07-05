# AI-Powered Nutrition Agent 🥗

An intelligent nutrition assistant built with **Python Flask** and **IBM Watsonx.ai (Granite 3)**.  
Features a rich, responsive chat UI, 7-day meal planner, calorie analyser, BMI/TDEE calculator, family diet plans, and Indian food support — all customisable through a single `AGENT_INSTRUCTIONS` block in `app.py`.

---

## Features

| Feature | Description |
|---|---|
| 🤖 AI Chat | Conversational nutrition advisor powered by IBM Granite |
| 🗓️ Meal Planner | 7-day personalised nutrition plans |
| 🔢 Calorie Analyser | Macro + micro breakdown for any meal |
| ⚖️ BMI & TDEE | Harris-Benedict formula with activity multipliers |
| 🍽️ Meal Ideas | 5 AI-generated meal suggestions with recipes |
| 👨‍👩‍👧 Family Plan | Unified + individual plans for multiple members |
| 🌙 Dark Mode | Full dark/light theme toggle |
| 📱 Mobile | Fully responsive Bootstrap 5 layout |
| 🇮🇳 Indian Food | Biased toward Indian cuisine when enabled |

---

## Project Structure

```
nutrition_agent/
├── app.py                  ← Flask backend + AGENT_INSTRUCTIONS
├── requirements.txt
├── .env.example            ← Copy to .env and fill credentials
├── .gitignore
├── README.md
└── templates/
    └── index.html          ← Full frontend (Bootstrap + vanilla JS)
```

---

## Quick Start

### 1. Clone / Download
```bash
git clone https://github.com/chinthanashivkumar/nutrition-agent.git
cd nutrition-agent
```

### 2. Create Virtual Environment
```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure Environment
```bash
cp .env.example .env
```

Edit `.env` with your credentials:
```
IBM_CLOUD_API_KEY=your_ibm_cloud_api_key
WATSONX_PROJECT_ID=your_watsonx_project_id
WATSONX_URL=https://us-south.ml.cloud.ibm.com
FLASK_SECRET_KEY=some-random-secret
FLASK_DEBUG=false
PORT=5000
```

#### Getting IBM Credentials
| Credential | Where to find |
|---|---|
| `IBM_CLOUD_API_KEY` | [IBM Cloud Console](https://cloud.ibm.com) → **Manage → Access → API Keys** → Create |
| `WATSONX_PROJECT_ID` | [Watsonx.ai](https://dataplatform.cloud.ibm.com) → **Projects** → your project → **Manage** tab → Project ID |
| `WATSONX_URL` | Depends on your region: `us-south`, `eu-de`, `eu-gb`, `jp-tok`, `au-syd` |

### 5. Run Development Server
```bash
python app.py
```

Open http://localhost:5000 in your browser.

---

## Customising the Agent

Open `app.py` and look for the **`AGENT_INSTRUCTIONS`** section near the top:

```python
# ── AGENT INSTRUCTIONS ────────────────────────────────────────────────────

AGENT_NAME             = "NutriBot"
AGENT_TONE             = "friendly, empathetic, and professional"
DEFAULT_DIET_MODE      = "balanced"          # vegetarian / vegan / keto …
INDIAN_FOOD_PREFERENCE = True                # bias toward Indian cuisine
RESPONSE_LANGUAGE      = "English"           # Hindi / Tamil / Bengali …
USE_EMOJIS_IN_RESPONSES = True
WATSONX_MODEL_ID       = "ibm/granite-3-3-8b-instruct"
```

All other behaviour (safety rules, response format, Watsonx parameters) is also configurable without touching any other part of the code.

---

## API Reference

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | Serve the frontend |
| POST | `/api/chat` | Conversational AI chat |
| POST | `/api/nutrition-plan` | Generate 7-day meal plan |
| POST | `/api/calorie-analysis` | Analyse food calories & macros |
| POST | `/api/family-plan` | Family-wide diet recommendations |
| POST | `/api/meal-suggestions` | Quick meal ideas |
| POST | `/api/bmi` | Calculate BMI |
| POST | `/api/tdee` | Calculate BMR & TDEE |
| GET | `/api/health` | Service health check |

---

## Production Deployment

### Option A — Gunicorn (Linux/macOS)
```bash
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

### Option B — Docker
```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 5000
CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:5000", "app:app"]
```
```bash
docker build -t nutribot .
docker run -p 5000:5000 --env-file .env nutribot
```

### Option C — IBM Code Engine
```bash
ibmcloud ce application create \
  --name nutribot \
  --image icr.io/your-namespace/nutribot:latest \
  --port 5000 \
  --env-from-secret nutribot-secrets
```

### Option D — Render / Railway / Fly.io
Set all `.env` variables as environment variables in the platform dashboard, then deploy the repo directly.

---

## Security Notes

- **Never commit `.env`** — it is in `.gitignore`
- Rotate your `FLASK_SECRET_KEY` in production
- Enable HTTPS in front of Gunicorn (use Nginx or a platform proxy)
- Consider rate-limiting `/api/chat` in production

---

## Supported Watsonx Models

| Model ID | Notes |
|---|---|
| `ibm/granite-3-3-8b-instruct` | Default — fast, instruction-tuned |
| `ibm/granite-3-8b-instruct` | Stable production model |
| `ibm/granite-13b-instruct-v2` | Larger, higher quality |
| `meta-llama/llama-3-70b-instruct` | Open-source alternative |

Change `WATSONX_MODEL_ID` in `AGENT_INSTRUCTIONS` to switch models.

---

## License
MIT — free for commercial and non-commercial use.
