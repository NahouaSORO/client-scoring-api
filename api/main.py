from fastapi import FastAPI
from pydantic import BaseModel
import joblib
from pathlib import Path
from typing import Literal

from src.predict import predict_risk, model, scaler, feature_columns, numeric_cols

BASE_DIR = Path(__file__).resolve().parent.parent
app = FastAPI(title="Credit Risk Prediction")

# Chargement du modèle
model = joblib.load(BASE_DIR / "models" / "xgb_model.pkl")

# Définition du schéma ou structure des requêtes
class ClientInput(BaseModel):
    person_age: int
    person_income: float
    person_home_ownership: Literal['RENT', 'OWN', 'MORTGAGE', 'OTHER']
    person_emp_length: int
    loan_intent: Literal['PERSONAL', 'EDUCATION', 'MEDICAL', 'VENTURE', 'HOMEIMPROVEMENT', 'DEBTCONSOLIDATION']
    loan_grade: Literal['A', 'B', 'C', 'D', 'E', 'F', 'G']
    loan_amnt: float
    loan_int_rate: float
    loan_percent_income: float
    cb_person_default_on_file: Literal['Y', 'N']
    cb_person_cred_hist_length: int

@app.get("/")
def root():
    return {"message": "Bienvenue sur l'API de Scoring Crédit. Utilisez /docs pour la documentation."}

@app.post("/predict")
def predict(data: ClientInput):
    client_dict = data.dict()
    result = predict_risk(client_dict)

    return result 

@app.get("/health")
def health():
    checks = {
        "api": "ok",
        "model_loaded": model is not None,
        "scaler_loaded": scaler is not None,
        "feature_columns_count": len(feature_columns),
        "numeric_cols_count": len(numeric_cols)
    }
    return checks

#D:
#cd client-scoring-api
#.venv\Scripts\activate.ps1

#uvicorn api.main:app --reload
#http://127.0.0.1:8000/docs
    

 