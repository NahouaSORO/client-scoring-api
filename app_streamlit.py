import streamlit as st
import requests

st.set_page_config(page_title="Simulateur de risque de crédit")
st.title("Simulateur de risque de crédit")
st.write("Remplissez les informations du client pour évaluer son risque de défaut de paiement.")

API_URL = "http://127.0.0.1:8000/predict"

with st.form("client_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        person_age = st.number_input(
            "Âge", min_value=18, max_value=100, value=None, 
            placeholder="Ex : 30"
        )
        person_income = st.number_input(
            "Revenu annuel", min_value=0, value=None, 
            placeholder="Ex : 45000"
        )
        person_home_ownership = st.selectbox(
            "Statut logement", 
            ['Sélectionner', 'RENT', 'OWN', 'MORTGAGE', 'OTHER']
        )
        person_emp_length = st.number_input(
            "Ancienneté professionnelle (années)", min_value=0, value=None, 
            placeholder="Ex : 3"
        )
        loan_intent = st.selectbox(
            "Motif du prêt",
            ['Sélectionner', 'PERSONAL', 'EDUCATION', 'MEDICAL', 'VENTURE', 
             'HOMEIMPROVEMENT', 'DEBTCONSOLIDATION']
        )
        loan_grade = st.selectbox(
            "Grade du prêt", 
            ['Sélectionner', 'A', 'B', 'C', 'D', 'E', 'F', 'G']
        )

    with col2:
        loan_amnt = st.number_input(
            "Montant du prêt", min_value=0, value=None, 
            placeholder="Ex : 8000"
        )
        loan_int_rate = st.number_input(
            "Taux d'intérêt (%)", min_value=0.0, value=None, 
            placeholder="Ex : 11.5"
        )
        loan_percent_income = st.number_input(
            "Ratio prêt / revenu", min_value=0.0, max_value=1.0, value=None, 
            placeholder="Ex : 0.10"
        )
        cb_person_default_on_file = st.selectbox(
            "Antécédent de défaut", 
            ['Sélectionner', 'Y', 'N']
        )
        cb_person_cred_hist_length = st.number_input(
            "Ancienneté historique de crédit (années)", min_value=0, value=None, 
            placeholder="Ex : 3"
        )

    submitted = st.form_submit_button("Évaluer le risque")

if submitted:
    # --- Validation : vérifie qu'aucun champ n'a été laissé vide ou non sélectionné ---
    champs_numeriques = {
        "Âge": person_age,
        "Revenu annuel": person_income,
        "Ancienneté professionnelle": person_emp_length,
        "Montant du prêt": loan_amnt,
        "Taux d'intérêt": loan_int_rate,
        "Ratio prêt / revenu": loan_percent_income,
        "Ancienneté historique de crédit": cb_person_cred_hist_length
    }
    champs_categoriels = {
        "Statut logement": person_home_ownership,
        "Motif du prêt": loan_intent,
        "Grade du prêt": loan_grade,
        "Antécédent de défaut": cb_person_default_on_file
    }

    manquants = [nom for nom, val in champs_numeriques.items() if val is None]
    manquants += [nom for nom, val in champs_categoriels.items() if val == 'Sélectionner']

    if manquants:
        st.warning(f"Merci de compléter les champs suivants : {', '.join(manquants)}")
    else:
        payload = {
            "person_age": person_age,
            "person_income": person_income,
            "person_home_ownership": person_home_ownership,
            "person_emp_length": person_emp_length,
            "loan_intent": loan_intent,
            "loan_grade": loan_grade,
            "loan_amnt": loan_amnt,
            "loan_int_rate": loan_int_rate,
            "loan_percent_income": loan_percent_income,
            "cb_person_default_on_file": cb_person_default_on_file,
            "cb_person_cred_hist_length": cb_person_cred_hist_length
        }

        try:
            response = requests.post(API_URL, json=payload, timeout=10)

            if response.status_code == 200:
                result = response.json()

                st.subheader("Résultat de l'évaluation")
                proba = result['probability_default']
                st.metric("Probabilité de défaut de paiement", f"{proba * 100:.2f} %")

                if "élevé" in result['prediction']:
                    st.error(f"⚠️ {result['prediction']}")
                else:
                    st.success(f"✅ {result['prediction']}")
        except requests.exceptions.ConnectionError:
            st.error("Impossible de contacter l'API. Vérifie qu'elle est bien lancée (uvicorn).")


            