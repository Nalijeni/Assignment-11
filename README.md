# Week 11 — Car Sales Forecast (XGBoost + FastAPI)

WHAT THIS DOES
- Trains an XGBoost regressor on monthly car sales using 6 lag features.
- Saves model to model/model.json and feature order to model/features.json.
- Serves a FastAPI with routes:
  * GET /health — health check
  * GET / — serves frontend/index.html
  * POST /predict — body: { "features": [lag_1, ..., lag_6] }

LOCAL RUN (Python)
1) pip install -r requirements.txt
2) uvicorn app.main:app --host 0.0.0.0 --port 8000
Open http://localhost:8000

DEPLOY ON RENDER
1) Push this repo to GitHub.
2) Create New Web Service on Render and connect the repo.
3) Build Command: (leave empty)
4) Start Command: uvicorn app.main:app --host 0.0.0.0 --port $PORT
(Also provided in Procfile.)

FILE LAYOUT
app/main.py
frontend/index.html
model/model.json
model/features.json
requirements.txt
Procfile
README.md
