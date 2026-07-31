# Importation des bibliothèques nécessaires
import pandas as pd
import json
import joblib
import xgboost as xgb
from matplotlib import pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix,roc_auc_score, ConfusionMatrixDisplay
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from preprocess import clean_data, engineer_features, encode_categoricals

# Configuration des chemins
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "raw" / "credit_risk.csv"
MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir( exist_ok=True)

def load_data(path):
    """Charge les données depuis un fichier CSV."""
    df = pd.read_csv(path)
    return df

def train_model():
    # Chargement des données
    df = load_data(DATA_PATH)
    print(f"\nForme initiale du dataset : {df.shape}")

    # Prétraitement des données (preprocessing)
    df = clean_data(df)
    df = engineer_features(df)
    df = encode_categoricals(df)
    print(df.columns.tolist())
    print(f"Forme du dataset après preprocessing complet : {df.shape}")

    # --- Sauvegarde du dataset "processed" (nettoyé + encodé) ---
    PROCESSED_DIR = BASE_DIR / "data" / "processed"
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    # parents=True : crée aussi le dossier "data" s'il n'existe pas déjà
    # exist_ok=True : ne plante pas si le dossier existe déjà

    df.to_csv(PROCESSED_DIR / "credit_risk_processed.csv", index=False)
    # index=False : évite d'ajouter une colonne "Unnamed: 0" inutile au rechargement
   # print(f"Dataset processed sauvegardé dans : {PROCESSED_DIR / 'credit_risk_processed.csv'}")

    # Séparation des features et de la cible
    X = df.drop(columns=['loan_status'])
    y = df['loan_status'] # On retire la cible (loan_status) des features

    # Sauvegarde de la liste avant le split
    feature_columns = X.columns.tolist()
    with open(MODEL_DIR / "feature_columns.json", "w") as f:
        json.dump(feature_columns, f)
    #print(f"Colonnes sauvegardées : {len(feature_columns)}")

    # Séparation en ensembles d'entraînement et de test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42,stratify=y)
    print(f"Train : {X_train.shape}, Test : {X_test.shape}")


    # Normalisation des features numériques
    numeric_cols = ['person_age', 'person_income', 'person_emp_length', 'loan_amnt', 
                    'loan_int_rate', 'loan_percent_income', 'cb_person_cred_hist_length',
                    'person_income_log', 'loan_grade']
    
    scaler = StandardScaler()
    X_train[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
    X_test[numeric_cols] = scaler.transform(X_test[numeric_cols])

    # Sauvegarde de la liste des colonnes numériques
    with open(MODEL_DIR / "numeric_cols.json", "w") as f:
        json.dump(numeric_cols, f)
    
    # Gestion du déséquilibre des classes 
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum() # le ratio de rééquilibrage des classes
    print(f"scale_pos_weight : {scale_pos_weight:.4f}")

    # Entraînement du modèle 
    model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.1,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        eval_metric='logloss'
        )
    model.fit(X_train, y_train)
    print("Modèle entraîné avec succès.")

    # Évaluation du modèle
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    print("\n=== CLASSIFICATION REPORT ===")
    print(classification_report(y_test, y_pred, target_names=['Aucun défaut de paiement (0)', 'Défaut de paiement (1)']))
    auc = roc_auc_score(y_test, y_pred_proba) # La métrique globale de qualité du modèle
    print(f"AUC-ROC : {auc:.4f}")

    # --- Matrice de confusion ---
    print("\n=== MATRICE DE CONFUSION  ===")
    cm_xgb = confusion_matrix(y_test, y_pred)
    disp_xgb = ConfusionMatrixDisplay(confusion_matrix=cm_xgb, display_labels=['Aucun défaut de paiement', 'Défaut de paiement'])
    disp_xgb.plot(cmap='Greens', values_format='d')
    plt.title('Matrice de confusion ')
    plt.show()

    # Sauvegarde du modèle et du scaler
    joblib.dump(model, MODEL_DIR / "xgb_model.pkl")
    joblib.dump(scaler, MODEL_DIR / "scaler.pkl")
    print(f"\nModèle et scaler sauvegardés avec succès ")


if __name__ == "__main__":
    train_model()
