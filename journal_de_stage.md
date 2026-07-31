# Journal de stage — Moteur de scoring client par ML exposé en API

---

## Séance 1 — Mise en place de l'environnement de travail

**Ce qui a été fait :**
- Création du dossier projet `client-scoring-api` sur `D:\`
- Création d'un environnement virtuel Python avec `python -m venv .venv`
- Installation des packages : `jupyter`, `ipykernel`, `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`
- Installation d'extensions VS Code : **Jupyter** (Microsoft)
- Création du premier notebook `00_test.ipynb` et test de bon fonctionnement
- Création du second notebook `01_exploration.ipynb` et test de bon fonctionnement

**Pourquoi :**
- Un environnement virtuel isole les packages du projet pour éviter les conflits avec d'autres projets Python sur la machine, et rend le projet reproductible (on peut recréer exactement le même environnement ailleurs).
- VS Code + extension Jupyter permet de travailler avec des notebooks sans changer d'application, en gardant le même éditeur pour tout le projet.
- Les notebooks (`.ipynb`) sont adaptés à la phase d'exploration de données car ils permettent d'exécuter le code par petits blocs (cellules) et de voir les résultats (tableaux, graphiques) directement sous le code, sans tout relancer à chaque fois.

**Vocabulaire appris :**
- **Kernel** : le moteur Python qui exécute le code du notebook en arrière-plan.
- **Environnement virtuel (venv)** : un espace Python isolé, propre à un projet, avec ses propres versions de packages.
- **Notebook** : fichier mélangeant code, texte et résultats, organisé en cellules exécutables.

**Erreurs rencontrées et solutions :**
- `rmdir /s` ne fonctionne pas sous PowerShell (syntaxe CMD) → utiliser `Remove-Item -Recurse -Force` à la place.
- Une commande `venv` interrompue par erreur (Ctrl+C) a laissé un dossier `venv` incomplet → suppression puis recréation propre en attendant la fin du processus sans interruption.

**Dataset retenu :**
- Credit Risk Dataset (32 581 lignes, 12 colonnes), récupéré via une copie GitHub du dataset Kaggle `laotse/credit-risk-dataset` pour éviter la contrainte de compte Kaggle. Colonnes en clair, cible `loan_status` déjà encodée en 0/1.
- Lien : https://www.kaggle.com/datasets/laotse/credit-risk-dataset

**Description des variables :**

| Variable   | Description |
- `person_age`: Âge du client 
- `person_income` : Revenu annuel du client 
- `person_home_ownership` : Statut du logement (locataire, propriétaire, hypothèque...) 
- `person_emp_length` : Ancienneté professionnelle (en années) 
- `loan_intent` : Motif du prêt (éducation, médical, personnel...) 
- `loan_grade` : Note de qualité de crédit attribuée au prêt (A à G, comme une note de risque) 
- `loan_amnt` : Montant du prêt demandé 
- `loan_int_rate` : Taux d'intérêt appliqué au prêt 
- `loan_status` : **Variable cible** : défaut de paiement (1) ou non (0) 
- `loan_percent_income` : Part du revenu que représente le prêt 
- `cb_person_default_on_file` : Défaut de paiement déjà survenu par le passé (Y/N) 
- `cb_person_cred_hist_length` : Durée de l'historique de crédit du client (en années) 

---

## Séance 2 — Exploration des données (EDA)

**Objectif :** se familiariser avec le pipeline complet (exploration → modèle → API) sur un cas réel.

**Distribution de la variable cible :**
- 78,2 % de non-défaut (0), 21,8 % de défaut (1) sur 32 581 clients.
- Déséquilibre modéré à prendre en compte lors du choix des métriques d'évaluation (AUC-ROC plutôt que l'accuracy seule) et lors du split train/test (utilisation de `stratify` pour préserver la proportion).

**Qualité des données :**
- Statistiques descriptives (`describe()`)
- Détection des valeurs manquantes : `person_emp_length` (895, ~2.7 %), `loan_int_rate` (3 116, ~9.6 %)
- Détection des doublons
- Analyse des distributions (histogrammes) des variables numériques
- Identification des outliers : `person_age` max 144, `person_emp_length` max 123, `person_income` très asymétrique (max 6 000 000)

**Analyse bivariée (relation avec `loan_status`) :**
- Boxplots numériques croisés avec la cible
- Tests statistiques (Mann-Whitney + effect size / rank-biserial) pour confirmer objectivement le pouvoir discriminant, plutôt que de se fier uniquement à la p-value (toutes significatives à p<0.05 vu la taille de l'échantillon, donc peu informatif seul)
- Tableaux de contingence pour les variables catégorielles
- Test du Chi² / Cramér's V pour les catégorielles

**Corrélations :**
- Matrice de corrélation entre variables numériques
- Multicolinéarité détectée : `person_age` / `cb_person_cred_hist_length` (0.88) → décision : garder les deux
- Quasi-duplication détectée : `loan_grade` / `loan_int_rate` (0.94) → décision : garder les deux, validée plus tard par un test de robustesse sur le modèle final

**Nettoyage :**
- Traitement des outliers : `person_age` ≤ 100 (seuil validé avec le maître de stage), `person_emp_length` ≤ 50
- Imputation `loan_int_rate` (médiane groupée par `loan_grade`, cohérent avec leur forte corrélation)
- Imputation `person_emp_length` (médiane simple)

**Feature Engineering :**
- `person_income_log` (log1p du revenu, pour corriger l'asymétrie de la distribution)

**Encodage :**
- `cb_person_default_on_file` → binaire (Y/N → 1/0)
- `loan_grade` → ordinale (A→0 … G→6)
- `person_home_ownership`, `loan_intent` → One-Hot Encoding

---

## Séance 3 — Modélisation

**Deux modèles comparés :**

| Métrique | Régression logistique | XGBoost |
|------------|----------------|------------  |
| AUC-ROC         | 0.868     | **0.951**    |
| Recall (Défaut) | 0.80      | 0.80–0.81    |
| Precision (Défaut) | 0.51   | 0.81–0.83    |

**Modèle retenu : XGBoost**, pour son AUC nettement supérieur et sa bien meilleure precision à recall équivalent (division par ~4 du nombre de faux positifs par rapport à la régression logistique).

**Lecture de la matrice de confusion (rappel des définitions) :**
- Vrai Négatif : prédit "Remboursé" ET c'était vraiment "Remboursé" (bonne prédiction)
- Faux Positif : prédit "Défaut" MAIS c'était en fait "Remboursé" (fausse alerte)
- Faux Négatif : prédit "Remboursé" MAIS c'était en fait "Défaut" (raté grave, on manque un vrai risque)
- Vrai Positif : prédit "Défaut" ET c'était vraiment "Défaut" (bonne détection)

**Courbe ROC :**
- fpr (False Positive Rate) = taux de fausses alertes, en abscisse
- tpr (True Positive Rate) = taux de vraies détections (= recall), en ordonnée
- thresholds = les différents seuils de décision testés

**Test de robustesse — risque de fuite de données (`loan_grade`) :**

> *"Bien que `loan_grade` soit la variable la plus importante du modèle et présente un risque théorique de circularité (elle est directement corrélée au taux d'intérêt, r=0.94, et progresse de façon quasi-déterministe avec le taux de défaut), un test de robustesse consistant à retirer cette variable montre que la performance du modèle ne chute que très marginalement (AUC : 0.9509 → 0.9465, soit -0.0043). Cela suggère que le modèle est capable de retrouver un pouvoir prédictif quasi équivalent à partir des autres variables disponibles, notamment `loan_int_rate` qui porte une information similaire. La décision a donc été prise de conserver `loan_grade` dans le modèle final, tout en documentant cette analyse de sensibilité."*

---

## Séance 4 — Passage du notebook au code industrialisé (`preprocess.py`, `train.py`)

**Objectif de la séance :** transitionner d'un environnement d'exploration (notebook) vers un code modulaire et robuste, prêt à être exposé via une API.

### 1. Refactoring et modularité

- **Centralisation du preprocessing** : sécurisation de l'importation des fonctions de nettoyage et d'ingénierie des variables depuis `preprocess.py` (`clean_data`, `engineer_features`, `encode_categoricals`). Évite toute duplication de code et garantit que l'API utilisera exactement la même logique que l'entraînement.
- **Encapsulation** : tout le pipeline d'entraînement encapsulé dans une fonction `train_model()`, facilitant de futurs réentraînements.
- Ajout de `align_columns()` et `full_preprocessing_pipeline()` dans `preprocess.py`, nécessaires pour traiter correctement un **client unique** (cas d'usage de l'API), différent du traitement sur dataset complet.

### 2. Résolution d'une anomalie majeure — bug de l'encodage (`drop_first`)

- **Problème** : une incohérence dans l'utilisation de `drop_first=True` (18 colonnes) vs `drop_first=False` (20 colonnes) risquait de faire planter le modèle lors de la prédiction sur un client unique — les catégories de référence exclues par défaut (ex : `MORTGAGE`, `DEBTCONSOLIDATION`) disparaissant complètement pour un client n'appartenant qu'à une seule catégorie.
- **Détection** : test avec un client fictif ayant `loan_intent='VENTURE'` et `person_home_ownership='OWN'` — ces colonnes dummy restaient à `0` malgré le vrai profil du client, faussant silencieusement la prédiction (probabilité passée de 0.1965 à 0.0001 après correction, pour un même client).
- **Solution** : fixation définitive de l'encodage sur `drop_first=False` (20 colonnes), combinée à `align_columns()` qui réaligne les colonnes d'un client unique sur la liste exacte utilisée à l'entraînement (`reindex` avec `fill_value=0`).

### 3. Résultats de l'entraînement final

- Volume de données : 32 581 lignes initiales → 31 679 lignes après preprocessing.
- Configuration du déséquilibre : `scale_pos_weight = 3.6416` (calculé dynamiquement sur le train set).

**Performance du modèle :**

| Métrique | Classe 0 (Remboursé) | Classe 1 (Défaut) | Global |
|---|---|---|---|
| Précision | 0.95 | 0.83 | Précision globale : 92 % |
| Rappel | 0.96 | 0.81 | AUC-ROC : 0.9510 |
| F1-Score | 0.95 | 0.82 | — |

**Artefacts générés et sauvegardés (`models/`) :**
- `xgb_model.pkl` — le modèle entraîné
- `scaler.pkl` — le standardiseur entraîné sur le train
- `feature_columns.json` — la liste ordonnée des 20 colonnes attendues
- `numeric_cols.json` — la liste des colonnes numériques à standardiser

**Bonne pratique retenue — reproductibilité :**
- `train.py` est conçu pour être relancé à volonté et regénérer systématiquement tous ses artefacts (modèle, scaler, colonnes, dataset processed) — aucune étape de sauvegarde n'est retirée après un premier succès, pour garantir que le script reste toujours reproductible à l'identique.

---

## Séance 5 — `predict.py` : prédiction sur un client unique

**Ce qui a été fait :**
- Écriture de `predict.py` : chargement des artefacts (`model`, `scaler`, `feature_columns`, `numeric_cols`) **une seule fois**, au niveau du module (pas à chaque appel, pour des raisons de performance).
- Fonction `predict_risk(client_data, threshold=0.5)` : applique `full_preprocessing_pipeline()` puis retourne le résultat de la prédiction.
- Validation par deux profils de test contrastés (client "sain" : propriétaire, grade B, faible ratio prêt/revenu vs client "à risque" : locataire, grade F, taux élevé, antécédent de défaut) — méthode dite des "cas limites" (edge cases), permettant de vérifier visuellement que le modèle réagit de façon cohérente sans ambiguïté d'interprétation.
- Résultats obtenus : 0.0001 (client sain) vs 0.9999 (client à risque) — confirme que le pipeline complet (preprocessing + modèle) fonctionne correctement de bout en bout.

**Format de la réponse retenu :**
- `probability_default` : probabilité brute (float, entre 0 et 1)
- `prediction` : phrase interprétable directement par un utilisateur non spécialiste (`"Risque de défaut faible"` / `"Risque de défaut élevé"`), plutôt qu'un simple entier 0/1 — décision prise pour rendre l'API utilisable sans connaissance du domaine.

**Erreur rencontrée et résolue :**
- Après une modification de la structure de retour de `predict_risk()`, l'interface Streamlit plantait avec `KeyError: 'label'` — la clé attendue par le frontend ne correspondait plus à la structure réellement retournée par l'API. Correction par cohérence entre les deux fichiers.

---

## Séance 6 — API FastAPI (`api/main.py`)

**Ce qui a été fait :**
- Mise en place de FastAPI, avec un schéma de requête (`ClientInput`, basé sur Pydantic) et un endpoint `POST /predict`.
- Endpoint `GET /health`, vérifiant que le modèle, le scaler et les fichiers de colonnes sont bien chargés en mémoire (pas seulement que le serveur répond).
- Passage de `src/` et `api/` en véritables **packages Python** (ajout de fichiers `__init__.py` vides), pour fiabiliser les imports entre modules, notamment sous le processus de rechargement automatique d'uvicorn (`--reload`).

**Pourquoi FastAPI :**
- Validation automatique des données entrantes (types ET valeurs autorisées via `Literal`), avec messages d'erreur clairs (HTTP 422) sans code de vérification manuel.
- Documentation interactive générée automatiquement (Swagger, `/docs`), utile pour tester sans écrire de client HTTP.

**Erreurs rencontrées et solutions :**
- `ModuleNotFoundError: No module named 'predict'` puis `'src'` puis `'preprocess'` : chaîne d'erreurs liées à des imports relatifs incohérents entre le mode "notebook/script isolé" et le mode "package importé par l'API". Résolu en généralisant les imports absolus (`from src.preprocess import ...`) dans tous les fichiers du package `src`, combinés aux `__init__.py`.
- `Literal` non reconnu (`NameError`) : oubli d'import (`from typing import Literal`) — les types de typage avancés de Python ne sont pas disponibles nativement, contrairement à `str`/`int`/`float`.
- **Bug d'inversion des `Literal`** : les listes de valeurs autorisées pour `person_home_ownership` et `loan_intent` avaient été inversées dans `ClientInput`, provoquant un rejet (422) de valeurs pourtant valides (ex : `"VENTURE"` refusé car validé avec la liste attendue pour le logement). Corrigé après inspection du détail de l'erreur JSON retournée par Pydantic.
- **Validation métier des catégories** : avant correction, une valeur hors nomenclature (ex : `loan_grade="H"`, `cb_person_default_on_file="O"`) était silencieusement transformée en `NaN` par le mapping (`.map()`), que XGBoost accepte nativement sans erreur — produisant une prédiction fantôme, non fiable, sans aucun signalement. Corrigé en contraignant chaque champ catégoriel à une liste `Literal` explicite dans `ClientInput`, rejetant toute valeur hors nomenclature avec une erreur 422 claire.

**Accès réseau local :**
- Par défaut, `uvicorn` n'écoute que sur `127.0.0.1` (la machine locale uniquement).
- Pour un accès depuis un autre appareil sur le même réseau Wi-Fi : `--host 0.0.0.0`, puis connexion via l'adresse IP locale (`ipconfig`) plutôt que `localhost`.

---

## Séance 7 — Interface utilisateur (Streamlit)

**Ce qui a été fait :**
- Écriture de `app_streamlit.py` : formulaire complet (11 champs), organisé en deux colonnes, correspondant exactement au schéma `ClientInput` de l'API.
- Champs vides par défaut avec `placeholder` indicatif (`value=None` + `placeholder="Ex : ..."`), plutôt qu'un pré-remplissage, pour éviter les erreurs de saisie par oubli et guider l'utilisateur sans lui imposer de valeurs.
- Validation manuelle avant envoi (vérification qu'aucun champ n'est resté vide ou sur `' Sélectionner '`), avec message listant précisément les champs manquants.
- Affichage du résultat avec code couleur (`st.success` / `st.error`) selon le niveau de risque.
- Gestion des erreurs de connexion (`try/except requests.exceptions.ConnectionError`) si l'API n'est pas démarrée.

**Pourquoi Streamlit :**
- Permet de créer une interface web fonctionnelle en pur Python, sans connaissances en HTML/CSS/JavaScript — adapté au contexte (projet de stage en data science, besoin d'une démo rapide et présentable).
- Alternative HTML/JS "vanilla" ou frameworks type React envisagés mais écartés : complexité disproportionnée par rapport au besoin (un formulaire à 11 champs avec un seul résultat affiché).

**Erreur rencontrée et résolue :**
- Traduction automatique du navigateur interférant avec les valeurs internes des listes déroulantes (`'PERSONAL'` transformé en `'PERSONNEL'` avant envoi à l'API) — désactivation de la traduction automatique de la page.

**Architecture retenue :**
- Streamlit et l'API tournent sur la même machine ; le code Python de `app_streamlit.py` (dont l'appel à l'API) s'exécute toujours côté serveur Streamlit, jamais dans le navigateur de l'utilisateur — donc `API_URL = "http://127.0.0.1:8000/predict"` reste valide même pour un utilisateur distant consultant la page depuis un autre appareil.

---

## Séance 8 — Documentation du projet

**Ce qui a été fait :**
- Mise à jour de `requirements.txt` via `python -m pip freeze`, pour inclure les dépendances ajoutées en cours de route (`streamlit`, `requests`), non présentes dans la version initiale.
- Rédaction du `README.md`, volontairement focalisé sur l'installation et l'utilisation de l'API/interface (et non sur la démarche d'exploration et de modélisation, qui reste dans ce journal et dans le rapport de stage).
- Distinction posée entre le rôle du **journal de stage** (chronologie, justifications, erreurs et apprentissages) et celui du **README** (documentation d'usage de l'état actuel du projet).

**Structure finale du projet :**

```
client-scoring-api/
├── data/
│   ├── raw/credit_risk.csv
│   └── processed/credit_risk_processed.csv
├── models/
│   ├── xgb_model.pkl
│   ├── scaler.pkl
│   ├── feature_columns.json
│   └── numeric_cols.json
├── notebooks/
├── src/
│   ├── __init__.py
│   ├── preprocess.py
│   ├── train.py
│   └── predict.py
├── api/
│   ├── __init__.py
│   └── main.py
├── app_streamlit.py
├── requirements.txt
└── README.md
```

---

## Prochaines étapes envisagées

- Sécurisation supplémentaire de `preprocess.py` (validation explicite en plus de celle déjà assurée par Pydantic, en défense en profondeur si les fonctions sont réutilisées hors API).
- Déploiement (Streamlit Community Cloud + hébergement de l'API), pour un accès sans dépendre d'une machine allumée en continu.
- Rédaction du rapport de stage final, structuré autour de : contexte métier, démarche EDA, choix et justification du modèle, analyse critique (limites, fuite de données, explicabilité), industrialisation, et pistes d'amélioration (autres modèles, monitoring en production).
