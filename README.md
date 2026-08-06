# Moteur de Scoring Client par Machine Learning exposé en API

API de scoring de risque de crédit : prédit la probabilité de défaut de paiement d'un client à partir de ses informations (revenu, historique de crédit, motif du prêt, etc.), via un modèle **XGBoost** entraîné sur le [Credit Risk Dataset](https://www.kaggle.com/datasets/laotse/credit-risk-dataset) (32 581 clients, 12 variables).

Une interface web (**Streamlit**) est fournie pour tester l'API via un formulaire, sans écrire de code.

---

## Installation

1. Décompresser le dossier (ou cloner le dépôt).
2. Ouvrir un terminal à la racine du projet.
3. Créer un environnement virtuel :
   ```powershell
   python -m venv .venv
   ```
4. Activer l'environnement :
   ```powershell
   .venv\Scripts\Activate.ps1      # PowerShell
   .venv\Scripts\activate.bat      # cmd.exe
   source .venv/bin/activate       # Mac/Linux
   ```
5. Installer les dépendances :
   ```powershell
   python -m pip install -r requirements.txt
   ```

---

## Lancer l'API

Depuis la *racine du projet*  :

```powershell
uvicorn api.main:app --reload
```

- API disponible sur : `http://127.0.0.1:8000`
- Documentation interactive (Swagger), générée automatiquement : `http://127.0.0.1:8000/docs`

---

## Endpoints de l'API

### `POST /predict`

Prédit le risque de défaut pour un client à partir de ses informations.

**Corps de la requête (JSON) :**

| Champ | Type | Valeurs acceptées |
|---|---|---|
| `person_age` | int | — |
| `person_income` | float | — |
| `person_home_ownership` | string | `RENT`, `OWN`, `MORTGAGE`, `OTHER` |
| `person_emp_length` | float | — |
| `loan_intent` | string | `PERSONAL`, `EDUCATION`, `MEDICAL`, `VENTURE`, `HOMEIMPROVEMENT`, `DEBTCONSOLIDATION` |
| `loan_grade` | string | `A`, `B`, `C`, `D`, `E`, `F`, `G` |
| `loan_amnt` | float | — |
| `loan_int_rate` | float | — |
| `loan_percent_income` | float | entre 0 et 1 |
| `cb_person_default_on_file` | string | `Y`, `N` |
| `cb_person_cred_hist_length` | int | — |

**Exemple de requête :**

```json
{
  "person_age": 28,
  "person_income": 45000,
  "person_home_ownership": "OWN",
  "person_emp_length": 3.0,
  "loan_intent": "VENTURE",
  "loan_grade": "B",
  "loan_amnt": 8000,
  "loan_int_rate": 11.5,
  "loan_percent_income": 0.18,
  "cb_person_default_on_file": "N",
  "cb_person_cred_hist_length": 4
}
```

**Exemple de réponse :**

```json
{
  "probability_default": 0.0001,
  "prediction": "Risque de défaut faible"
}
```

| Champ | Type | Description |
|---|---|---|
| `probability_default` | float | Probabilité de défaut estimée par le modèle (entre 0 et 1) |
| `prediction` | string | Interprétation lisible du résultat, exprimée en phrase (`"Risque de défaut faible"` ou `"Risque de défaut élevé"`), pensée pour un utilisateur non spécialiste du scoring |

Le seuil de décision par défaut est de **0.5** (probabilité ≥ 0.5 → `"Risque de défaut élevé"`), ajustable via le paramètre `threshold` de la fonction `predict_risk()`.

Toute valeur en dehors des catégories acceptées (ex : `loan_grade: "H"`) est rejetée avec une erreur **422**, détaillant précisément le champ fautif et les valeurs autorisées.

### `GET /health`

Vérifie que l'API et ses artefacts (modèle, scaler, colonnes) sont correctement chargés en mémoire.

**Exemple de réponse :**

```json
{
  "api": "ok",
  "model_loaded": true,
  "scaler_loaded": true,
  "feature_columns_count": 20,
  "numeric_cols_count": 9
}
```

## Lancer l'interface web (Streamlit)

Dans un **second terminal**, l'API devant rester active dans le premier :

```powershell
streamlit run app_streamlit.py
```

L'interface s'ouvre automatiquement sur `http://localhost:8501` avec un formulaire guidé (placeholders, validation des champs, listes déroulantes pour les variables catégorielles).

---

## Accès depuis un autre appareil sur le même réseau

```powershell
# Terminal 1 — API
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

# Terminal 2 — Interface
streamlit run app_streamlit.py --server.address 0.0.0.0 --server.port 8501
```

Trouver son adresse IP locale : `ipconfig` (ligne "Adresse IPv4"). Depuis un autre appareil connecté au même réseau Wi-Fi : `http://<adresse-ip>:8501`.

---

# Etapes de réalisation du projet
- Mise en place de l'environnement de travail
- Exploration de données (EDA)
- Modélisatiion
- Passage du notebook au code industrialisé (preprocess.py, train.py)
- predict.py : Prédiction sur un  client unique
- Réalisation de l'API avec FastAPI (api/main.py)
- Interface utilissateur (Streamlit)
- Documentation du projet
