# 🔍 TalentLensAI

ML-powered resume-to-role scorer and job market trend analyzer — built with TensorFlow, scikit-learn, and Streamlit. Includes full ML observability: prediction logging, input drift detection, outcome tracking, and a live calibration dashboard.

## Live Demo
👉 [talentlensai.streamlit.app](https://talentlensai-ssuzhnwjk7kvk69k9ofc7t.streamlit.app)

## What it does

### 🎯 Resume Scorer
Paste your skills, years of experience, and education level against a job posting. The TensorFlow model scores the match (0–100%) and highlights missing skills. Flags input drift if your profile is outside the training distribution.

### 📈 Market Trends
Select any combination of skills (Python, LLMs, RAG, PyTorch, etc.) and see predicted demand scores from 2020 through 2026. Built on a trend model trained on simulated Statcast-style growth curves for 18 in-demand tech skills.

### 📊 Observability Dashboard
Every prediction is logged to SQLite with inputs, output score, and optional outcome. The dashboard shows:
- Score distribution histogram
- Calibration curve (predicted score vs actual interview rate)
- Feature drift table (training baseline vs recent inputs)
- Full prediction log

## Tech Stack
- **TensorFlow / Keras** — two neural networks (regression)
- **scikit-learn** — preprocessing, train/test split
- **Streamlit** — interactive UI with dark purple theme
- **SQLite** — lightweight prediction logger
- **Python** — data generation, drift detection, model serving

## Models

| Model | Task | MAE |
|---|---|---|
| Match Scorer | Resume-to-role match (0–1) | 0.028 |
| Trend Predictor | Skill demand score (0–10) | 1.578 |

## Run locally

```bash
git clone https://github.com/Tatopapi3/TalentlensAI.git
cd TalentlensAI
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Train models (generates synthetic data + saves model files)
python train.py

# Launch app
streamlit run app.py
```

## Project Structure

```
talentlensai/
├── app.py                  # Streamlit UI
├── models/
│   ├── match_scorer.py     # Resume-to-role TF model
│   └── trend_predictor.py  # Job market trend TF model
├── core/
│   ├── logger.py           # SQLite prediction logger
│   └── drift.py            # Input drift detector
├── data/
│   └── generate_data.py    # Synthetic training data generator
├── train.py                # Train both models
└── requirements.txt
```
