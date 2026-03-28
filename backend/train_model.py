import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.preprocessing import LabelEncoder
import joblib
import os

def train():
    print("--- Starting Training Process ---")
    
    # 1. Load Datasets
    try:
        df_symptoms = pd.read_csv('data/dataset.csv')
        df_precautions = pd.read_csv('data/Disease precaution.csv')
        df_descriptions = pd.read_csv('data/symptom_Description.csv')
        print("All 3 CSV files loaded successfully.")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please ensure dataset.csv, Disease precaution.csv, and symptom_Description.csv are in backend/data/")
        return

    # 2. Data Cleaning
    for col in df_symptoms.columns:
        df_symptoms[col] = df_symptoms[col].astype(str).str.replace('_', ' ').str.strip()
    
    df_precautions['Disease'] = df_precautions['Disease'].astype(str).str.strip()
    df_descriptions['Disease'] = df_descriptions['Disease'].astype(str).str.strip()
    df_descriptions['Description'] = df_descriptions['Description'].astype(str).str.strip()

    # 3. Extract All Unique Symptoms (Vocabulary)
    symptom_cols = [col for col in df_symptoms.columns if 'Symptom' in col]
    all_symptoms_raw = df_symptoms[symptom_cols].values.flatten()
    unique_symptoms = list(set([s for s in all_symptoms_raw if pd.notna(s) and s.lower() != 'nan' and s != '']))
    unique_symptoms.sort()
    
    print(f"Vocabulary Size: {len(unique_symptoms)} unique symptoms.")

    # 4. Prepare Training Data (One-Hot Encoding)
    X = []
    y = df_symptoms['Disease'].values

    for index, row in df_symptoms.iterrows():
        current_symptoms = [str(s).strip() for s in row[symptom_cols].values if pd.notna(s) and s.lower() != 'nan']
        binary_vector = [1 if sym in current_symptoms else 0 for sym in unique_symptoms]
        X.append(binary_vector)

    X = np.array(X)
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    # 5. Train Voting Ensemble (Random Forest + Gradient Boosting)
    print("Training Voting Ensemble (RF + GradientBoosting)...")
    
    rf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    gb = GradientBoostingClassifier(n_estimators=150, learning_rate=0.1, max_depth=5, random_state=42)
    
    ensemble = VotingClassifier(
        estimators=[('rf', rf), ('gb', gb)],
        voting='soft'
    )
    ensemble.fit(X, y_encoded)
    print("Ensemble trained successfully.")
    
    # 6. Build Knowledge Dictionaries
    
    # A. Precautions Map
    precaution_map = {}
    for index, row in df_precautions.iterrows():
        disease = row['Disease']
        precs = [row[f'Precaution_{i}'] for i in range(1, 5) if pd.notna(row.get(f'Precaution_{i}'))]
        precs = [p.strip().capitalize() for p in precs]
        precaution_map[disease] = precs

    # B. Description Map
    description_map = dict(zip(df_descriptions['Disease'], df_descriptions['Description']))

    # 7. Save Everything
    artifacts = {
        "model": ensemble,
        "label_encoder": le,
        "symptoms_list": unique_symptoms,
        "precautions": precaution_map,
        "descriptions": description_map
    }
    
    os.makedirs('models', exist_ok=True)
    joblib.dump(artifacts, 'models/disease_model.pkl')
    print("Model trained and saved to backend/models/disease_model.pkl")

if __name__ == "__main__":
    train()