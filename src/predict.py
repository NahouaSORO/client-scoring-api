# Importation des bibliothèques nécessaires
import pandas as pd 
import json 
import joblib
from pathlib import Path

try:
    from src.preprocess import full_preprocessing_pipeline
except ModuleNotFoundError:
    from preprocess import full_preprocessing_pipeline

# Configuration des chemins
BASE_DIR = Path(__file__).resolve().parent.parent  
MODEL_DIR = BASE_DIR / "models"

# Chargement des artefacts du modèle
model = joblib.load(MODEL_DIR / "xgb_model.pkl")
scaler = joblib.load(MODEL_DIR / "scaler.pkl")

with open(MODEL_DIR / "feature_columns.json") as f:
    feature_columns = json.load(f)

with open(MODEL_DIR / "numeric_cols.json") as f:
    numeric_cols = json.load(f)

def predict_risk(client_data, threshold: float = 0.5):
    """
    Prédit le risque de défaut pour un client donné.
    Args:
        client_data (dict): Dictionnaire contenant les informations du client.
        threshold (float): Seuil pour classer le risque (par défaut 0.5).
    Returns:
        dict: Dictionnaire contenant la probabilité de défaut et la prédiction finale.
    """

    df_client = pd.DataFrame([client_data])
    df_client_processed = full_preprocessing_pipeline(df_client, scaler, feature_columns, numeric_cols)

    proba = model.predict_proba(df_client_processed)[:, 1][0]
    prediction = "Risque de défaut élevé" if proba >= threshold else "Risque de défaut faible"

    return {
        "probability_default": round(float(proba), 4),
        "prediction": prediction,     
    }

if __name__ == "__main__":
    BASE_DIR = Path(__file__).resolve().parent.parent

    model = joblib.load(BASE_DIR / "models" / "xgb_model.pkl")
    scaler = joblib.load(BASE_DIR / "models" / "scaler.pkl")

    with open(BASE_DIR / "models" / "feature_columns.json") as f:
        feature_columns = json.load(f)

    # Charger les colonnes numériques attendues
    with open(BASE_DIR / "models" / "numeric_cols.json") as f:
        numeric_cols = json.load(f)

    # Pour éviter un décalage entre la liste de colonnes stockée et
    # celles réellement utilisées par le modèle (par ex. catégories
    # absentes lors de l'entraînement), on force l'ordre de colonnes
    # attendu en se basant sur le modèle chargé.
    try:
        model_feature_names = model.get_booster().feature_names
        if model_feature_names is not None:
            feature_columns = list(model_feature_names)
    except Exception:
        # En cas d'échec, on garde la liste chargée depuis le JSON
        pass

    nouveau_client = {
        'person_age': 28,
        'person_income': 45000,
        'person_home_ownership': 'OWN',
        'person_emp_length': 3.0,
        'loan_intent': 'VENTURE',
        'loan_grade': 'F',
        'loan_amnt': 8000,
        'loan_int_rate': 11.5,
        'loan_percent_income': 0.18,
        'cb_person_default_on_file': 'N',
        'cb_person_cred_hist_length': 4
    }

    df_client = pd.DataFrame([nouveau_client])
    print("\n === Client brut === ")
    print(df_client)

    df_client_processed = full_preprocessing_pipeline(df_client, scaler, feature_columns, numeric_cols)

    print("\n === Client après preprocessing complet === ")
    print(df_client_processed)
    print(f"\nShape finale du dataset : {df_client_processed.shape}")
    print(f"\nColonnes : {df_client_processed.columns.tolist()}")

    proba = model.predict_proba(df_client_processed)[:, 1][0]
    prediction = int(proba >= 0.5)

    print(f"\nProbabilité de défaut de paiement : {proba:.4f}")
    print(f"Décision du modèle : {'Ce client présente un risque de défaut de paiement' if prediction == 1 else 'Aucun défaut de paiement'}")

