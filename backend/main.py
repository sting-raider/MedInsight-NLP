from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import joblib
import numpy as np
from duckduckgo_search import DDGS
import wikipedia
import re
import spacy

app = FastAPI(title="MedInsight API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Load Data ---
try:
    data = joblib.load('models/disease_model.pkl')
    model = data['model']
    le = data['label_encoder']
    symptoms_list = data['symptoms_list']
    precaution_map = data.get('precautions', {})
    description_map = data.get('descriptions', {})
    nlp = spacy.load("en_core_web_sm")
    print("System Loaded Successfully.")
except Exception as e:
    print(f"Error loading model: {e}")

class SymptomInput(BaseModel):
    text: str

# --- 1. DISEASE PREVALENCE MAP (The Fix) ---
# We categorize diseases to fix the "Rare Disease Bias"
# HIGH = Very Common (Flu, Allergy)
# MEDIUM = Specific but not life-threatening (Piles, Migraine)
# LOW = Rare or Severe (AIDS, Heart Attack) - Requires strong evidence
DISEASE_PREVALENCE = {
    # COMMON (Boost these)
    'Fungal infection': 'HIGH',
    'Allergy': 'HIGH',
    'GERD': 'HIGH',
    'Chronic cholestasis': 'MEDIUM',
    'Drug Reaction': 'MEDIUM',
    'Peptic ulcer diseae': 'MEDIUM',
    'AIDS': 'LOW', # Severe
    'Diabetes ': 'MEDIUM',
    'Gastroenteritis': 'HIGH',
    'Bronchial Asthma': 'MEDIUM',
    'Hypertension ': 'MEDIUM',
    'Migraine': 'HIGH',
    'Cervical spondylosis': 'MEDIUM',
    'Paralysis (brain hemorrhage)': 'LOW', # Severe
    'Jaundice': 'MEDIUM',
    'Malaria': 'MEDIUM',
    'Chicken pox': 'MEDIUM',
    'Dengue': 'MEDIUM',
    'Typhoid': 'MEDIUM',
    'hepatitis A': 'MEDIUM',
    'Hepatitis B': 'LOW',
    'Hepatitis C': 'LOW',
    'Hepatitis D': 'LOW',
    'Hepatitis E': 'LOW',
    'Alcoholic hepatitis': 'MEDIUM',
    'Tuberculosis': 'LOW',
    'Common Cold': 'HIGH',
    'Pneumonia': 'MEDIUM',
    'Dimorphic hemmorhoids(piles)': 'MEDIUM',
    'Heart attack': 'LOW', # Severe
    'Varicose veins': 'MEDIUM',
    'Hypothyroidism': 'MEDIUM',
    'Hyperthyroidism': 'MEDIUM',
    'Hypoglycemia': 'MEDIUM',
    'Osteoarthristis': 'MEDIUM',
    'Arthritis': 'MEDIUM',
    '(vertigo) Paroymsal Positional Vertigo': 'MEDIUM',
    'Acne': 'HIGH',
    'Urinary tract infection': 'HIGH',
    'Psoriasis': 'MEDIUM',
    'Impetigo': 'MEDIUM'
}

# --- 2. NLP Logic (Keep existing) ---
SYNONYM_MAP = {
    "rash": "skin_rash",
    "spots": "skin_rash",
    "eruption": "skin_rash",
    "redness": "skin_rash",
    "itching": "itching",
    "itch": "itching",
    "scratching": "itching",
    "shiver": "chills",
    "shivering": "chills",
    "cold": "chills",
    "joint pain": "joint_pain",
    "pain in joints": "joint_pain",
    "knee pain": "joint_pain",
    "stomach ache": "stomach_pain",
    "abdominal pain": "stomach_pain",
    "belly pain": "stomach_pain",
    "tummy hurt": "stomach_pain",
    "vomit": "vomiting",
    "puking": "vomiting",
    "throw up": "vomiting",
    "nauseous": "nausea",
    "sickness": "nausea",
    "tired": "fatigue",
    "fatigue": "fatigue",
    "weakness": "fatigue",
    "cough": "cough",
    "coughing": "cough",
    "fever": "high_fever",
    "temp": "high_fever",
    "hot": "high_fever",
    "headache": "headache",
    "head hurts": "headache",
    "migraine": "headache",
    "dizzy": "dizziness",
    "spinning": "dizziness",
}

def extract_symptoms(user_text):
    user_text = user_text.lower()
    user_text = re.sub(r'[^\w\s]', ' ', user_text)
    
    detected = set()

    # Synonyms
    for key, value in SYNONYM_MAP.items():
        if f" {key} " in f" {user_text} ": 
            detected.add(value)

    # Standard & Partial
    for symptom in symptoms_list:
        if symptom in user_text:
            detected.add(symptom)
        symptom_tokens = symptom.split()
        if len(symptom_tokens) > 1:
            if any(token in user_text for token in symptom_tokens if len(token) > 3): 
                 if "rash" in user_text and "rash" in symptom: detected.add("skin_rash")
                 if "fever" in user_text and "fever" in symptom: detected.add("high_fever")
                 if "stomach" in user_text and "stomach" in symptom: detected.add("stomach_pain")

    return list(detected)

# --- 3. Web Context (Wikipedia Fallback) ---
def fetch_web_context(disease_name):
    print(f"🔎 Fetching context for: {disease_name}")
    try:
        # Wikipedia is cleaner for definitions
        summary = wikipedia.summary(disease_name, sentences=2)
        return f"{summary} (Source: Wikipedia)"
    except:
        pass
    
    try:
        # DuckDuckGo as backup
        with DDGS() as ddgs:
            results = list(ddgs.text(f"{disease_name} medical summary", max_results=1))
            if results:
                return results[0]['body']
    except:
        pass
    return "No live context available."

@app.post("/predict")
async def predict(input_data: SymptomInput):
    user_symptoms = extract_symptoms(input_data.text)
    
    if not user_symptoms:
        return {
            "status": "failed", 
            "message": "No symptoms detected. Try terms like 'fever', 'cough', 'rash'."
        }

    # Vectorize
    input_vector = [1 if s in user_symptoms else 0 for s in symptoms_list]
    input_vector = np.array(input_vector).reshape(1, -1)

    # Raw Prediction
    probs = model.predict_proba(input_vector)[0]
    
    # --- 4. BAYESIAN RE-RANKING (The Magic Logic) ---
    
    # If the user gives vague input (< 3 symptoms), rely heavily on commonality
    is_vague_input = len(user_symptoms) < 3
    
    adjusted_probs = []
    for i, prob in enumerate(probs):
        disease = le.inverse_transform([i])[0]
        severity = DISEASE_PREVALENCE.get(disease, 'MEDIUM')
        
        # Apply Weights
        weight = 1.0
        
        if is_vague_input:
            if severity == 'HIGH':
                weight = 1.3  # Boost Common (Flu, Allergy)
            elif severity == 'LOW':
                weight = 0.5  # Penalize Severe (AIDS, Heart Attack)
        
        # Calculate new score
        new_score = prob * weight
        adjusted_probs.append((disease, new_score))
        
    # Sort by New Score
    adjusted_probs.sort(key=lambda x: x[1], reverse=True)
    
    # Take top 3
    top_results = adjusted_probs[:3]

    results = []
    for i, (disease, score) in enumerate(top_results):
        # Normalize score slightly for display (cap at 99%)
        display_conf = min(score, 0.99)
        
        if display_conf > 0.01:
            web_context = ""
            if i == 0: 
                web_context = fetch_web_context(disease)

            results.append({
                "disease": disease,
                "confidence": round(display_conf * 100, 2),
                "description": description_map.get(disease, "Definition unavailable."),
                "precautions": precaution_map.get(disease, []),
                "web_context": web_context
            })

    return {
        "status": "success",
        "extracted_symptoms": list(user_symptoms),
        "predictions": results
    }