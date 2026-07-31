# Importation des bibliothèques nécessaires 
import pandas as pd
import numpy as np 

# chargement du dataset
df = pd.read_csv("data/raw/credit_risk.csv")

def clean_data(df):
    df = df.copy()

    # Filtrage des valeurs abérantes 
    df = df[df['person_age'] <= 100]
    df = df[df['person_emp_length'] <= 80]

    # Traitements des valeurs manquantes (Imputation par la médiane)
    df['loan_int_rate']= df.groupby('loan_grade')['loan_int_rate'].transform(
        lambda x: x.fillna(x.median()))
    df['person_emp_length']= df['person_emp_length'].fillna(df['person_emp_length'].median())

    return df

def engineer_features(df):
    df = df.copy()

    # Création de nouvelles features
    df['person_income_log'] = np.log1p(df['person_income'])
    return df

def encode_categoricals(df):
    df = df.copy()

    # Encodage numérique A-G -> 0-6
    grade_order = {'A':0, 'B':1, 'C':2, 'D':3, 'E':4, 'F':5, 'G':6}
    df['loan_grade'] = df['loan_grade'].map(grade_order)

    # Encodage binaire Y/N -> 1/0
    df['cb_person_default_on_file'] = df['cb_person_default_on_file'].map({'Y': 1, 'N': 0})

    # Nominale : One-Hot Encoding (transforme les colonnes catégorielles en binaires)
    df = pd.get_dummies(df, columns=['person_home_ownership', 'loan_intent'])
    return df

def align_columns(df, expected_columns):
    """Aligne les colonnes sur la liste exacte utilisée à l'entraînement."""
    df = df.reindex(columns = expected_columns, fill_value = 0)
    return df

def full_preprocessing_pipeline(df, scaler, expected_columns, numeric_cols ):
    """
    Enchaîne toutes les étapes de preprocessing, de la donnée brute
    jusqu'au format final prêt à être injecté dans le modèle.
    """
    df = clean_data(df)
    df = engineer_features(df)
    df = encode_categoricals(df)
    df = align_columns(df, expected_columns)
    df[numeric_cols] = scaler.transform(df[numeric_cols])
    return df


