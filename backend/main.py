from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import joblib
import numpy as np
import re
import os
import difflib
import httpx
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = "llama-3.3-70b-versatile"
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"

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
    print("System Loaded Successfully.")
except Exception as e:
    print(f"Error loading model: {e}")

class SymptomInput(BaseModel):
    text: str


# --- Health Endpoint ---
@app.get("/health")
async def health():
    return {"status": "ok"}


# --- Disease Prevalence Map ---
DISEASE_PREVALENCE = {
    'Fungal infection': 'HIGH',
    'Allergy': 'HIGH',
    'GERD': 'HIGH',
    'Chronic cholestasis': 'MEDIUM',
    'Drug Reaction': 'MEDIUM',
    'Peptic ulcer diseae': 'MEDIUM',
    'AIDS': 'LOW',
    'Diabetes ': 'MEDIUM',
    'Gastroenteritis': 'HIGH',
    'Bronchial Asthma': 'MEDIUM',
    'Hypertension ': 'MEDIUM',
    'Migraine': 'HIGH',
    'Cervical spondylosis': 'MEDIUM',
    'Paralysis (brain hemorrhage)': 'LOW',
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
    'Heart attack': 'LOW',
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


# --- Expanded NLP Synonym Map ---
# IMPORTANT: Values MUST use spaces (not underscores) to match symptoms_list vocabulary.
# The regex strips apostrophes, so "can't" → "cant" — include both forms.
SYNONYM_MAP = {
    # Skin
    "rash": "skin rash", "spots": "skin rash", "eruption": "skin rash",
    "redness": "skin rash", "hives": "skin rash", "bumps on skin": "skin rash",
    "blotchy skin": "skin rash", "red patches": "skin rash",
    "pimples": "acne", "zits": "acne", "breakout": "acne",
    "flaky skin": "skin peeling", "peeling": "skin peeling", "skin peeling": "skin peeling",
    "yellow skin": "yellowish skin", "yellowish": "yellowish skin", "jaundiced": "yellowish skin",
    "bruising": "bruising", "bruises": "bruising", "purple spots": "bruising",

    # Itching
    "itching": "itching", "itch": "itching", "itchy": "itching",
    "scratching": "itching", "irritated skin": "itching",

    # Temperature / Fever
    "fever": "high fever", "temp": "high fever",
    "burning up": "high fever", "feverish": "high fever", "temperature": "high fever",
    "high temperature": "high fever", "feeling hot": "high fever",
    "mild fever": "mild fever", "low grade fever": "mild fever", "slight fever": "mild fever",
    "shiver": "chills", "shivering": "chills", "cold sweats": "chills",
    "freezing": "chills", "goosebumps": "chills",
    "sweating": "sweating", "sweaty": "sweating", "night sweats": "sweating",
    "perspiring": "sweating",

    # Pain
    "joint pain": "joint pain", "pain in joints": "joint pain",
    "knee pain": "knee pain", "achy joints": "joint pain", "stiff joints": "joint pain",
    "stomach ache": "stomach pain", "abdominal pain": "abdominal pain",
    "belly pain": "belly pain", "tummy hurts": "stomach pain",
    "tummy hurt": "stomach pain", "gut pain": "stomach pain",
    "headache": "headache", "head hurts": "headache", "migraine": "headache",
    "head pounding": "headache", "throbbing head": "headache",
    "chest pain": "chest pain", "chest hurts": "chest pain",
    "chest tightness": "chest pain", "pressure in chest": "chest pain",
    "back pain": "back pain", "lower back pain": "back pain",
    "backache": "back pain", "spine pain": "back pain",
    "body ache": "muscle pain", "body aches": "muscle pain",
    "muscle pain": "muscle pain", "sore muscles": "muscle pain",
    "body pain": "muscle pain", "aching all over": "muscle pain", "sore body": "muscle pain",
    "neck pain": "neck pain", "stiff neck": "stiff neck",
    "hip pain": "hip joint pain", "pain behind eyes": "pain behind the eyes",
    "eye pain": "pain behind the eyes",
    "cramps": "cramps", "muscle cramps": "cramps",

    # Digestive
    "vomit": "vomiting", "puking": "vomiting", "throw up": "vomiting",
    "throwing up": "vomiting", "retching": "vomiting", "being sick": "vomiting",
    "nauseous": "nausea", "nausea": "nausea", "queasy": "nausea",
    "sickness": "nausea", "sick feeling": "nausea", "feel sick": "nausea",
    "diarrhea": "diarrhoea", "diarrhoea": "diarrhoea", "loose stool": "diarrhoea",
    "runny stool": "diarrhoea", "watery stool": "diarrhoea", "the runs": "diarrhoea",
    "constipation": "constipation", "bloated": "distention of abdomen",
    "bloating": "distention of abdomen", "gassy": "distention of abdomen",
    "gas": "passage of gases", "flatulence": "passage of gases",
    "indigestion": "indigestion", "acid reflux": "acidity",
    "heartburn": "acidity", "acidic": "acidity", "burping": "acidity",
    "loss of appetite": "loss of appetite", "not hungry": "loss of appetite",
    "no appetite": "loss of appetite", "cant eat": "loss of appetite",
    "weight loss": "weight loss", "losing weight": "weight loss",
    "weight gain": "weight gain", "gaining weight": "weight gain",
    "bloody stool": "bloody stool", "blood in stool": "bloody stool",

    # Respiratory
    "cough": "cough", "coughing": "cough", "dry cough": "cough",
    "hacking cough": "cough", "persistent cough": "cough",
    "phlegm": "phlegm", "mucus": "phlegm", "sputum": "phlegm",
    "runny nose": "runny nose", "stuffy nose": "congestion",
    "nasal congestion": "congestion", "blocked nose": "congestion",
    "congested": "congestion", "stuffed up": "congestion",
    "sneezing": "continuous sneezing", "sneeze": "continuous sneezing",
    "cant stop sneezing": "continuous sneezing",
    "breathless": "breathlessness", "short of breath": "breathlessness",
    "cant breathe": "breathlessness", "difficulty breathing": "breathlessness",
    "wheezing": "breathlessness",
    "sinus": "sinus pressure", "sinusitis": "sinus pressure", "sinus pain": "sinus pressure",

    # Smell & Taste (commonly missed — especially for flu/COVID-like inputs)
    "cant smell": "loss of smell", "no smell": "loss of smell",
    "lost smell": "loss of smell", "anosmia": "loss of smell",
    "loss of smell": "loss of smell", "cant smell anything": "loss of smell",
    "lost my smell": "loss of smell", "smell gone": "loss of smell",
    "cant taste": "altered sensorium", "no taste": "altered sensorium",
    "lost taste": "altered sensorium", "ageusia": "altered sensorium",
    "loss of taste": "altered sensorium", "cant taste anything": "altered sensorium",
    "lost my taste": "altered sensorium", "taste gone": "altered sensorium",
    "food tasteless": "altered sensorium",

    # Throat / Mouth
    "sore throat": "throat irritation", "scratchy throat": "throat irritation",
    "throat hurts": "throat irritation", "hoarse voice": "throat irritation",
    "throat pain": "throat irritation", "throat irritation": "throat irritation",
    "raspy voice": "throat irritation", "itchy throat": "throat irritation",
    "swollen throat": "throat irritation", "burning throat": "throat irritation",
    "very sore throat": "throat irritation", "really sore throat": "throat irritation",
    "patches in throat": "patches in throat",
    "difficulty swallowing": "patches in throat",
    "hard to swallow": "patches in throat",
    "ulcers in mouth": "ulcers on tongue", "mouth sores": "ulcers on tongue",
    "bad breath": "foul smell of urine",

    # Fatigue / General
    "tired": "fatigue", "fatigue": "fatigue", "weakness": "fatigue",
    "exhausted": "fatigue", "no energy": "fatigue", "worn out": "fatigue",
    "run down": "fatigue", "wiped out": "fatigue",
    "lethargic": "lethargy", "sluggish": "lethargy", "drowsy": "lethargy",
    "restless": "restlessness", "cant sit still": "restlessness",
    "anxious": "anxiety", "anxiety": "anxiety", "nervous": "anxiety",
    "depressed": "depression", "sad": "depression", "hopeless": "depression",
    "mood swings": "mood swings", "moody": "mood swings", "irritable": "irritability",
    "malaise": "malaise", "unwell": "malaise", "feeling unwell": "malaise",
    "generally unwell": "malaise", "not feeling well": "malaise",

    # Head / Neuro
    "dizzy": "dizziness", "spinning": "dizziness", "vertigo": "dizziness",
    "lightheaded": "dizziness", "unsteady": "loss of balance",
    "balance issues": "loss of balance", "wobbly": "loss of balance",
    "blurry vision": "blurred and distorted vision",
    "blurred vision": "blurred and distorted vision",
    "cant see clearly": "blurred and distorted vision",
    "fainting": "coma", "passed out": "coma",
    "blackout": "coma", "fainted": "coma",
    "slurred speech": "slurred speech", "mumbling": "slurred speech",
    "cant concentrate": "lack of concentration", "brain fog": "lack of concentration",
    "confused": "altered sensorium",

    # Urinary
    "burning pee": "burning micturition", "painful urination": "burning micturition",
    "burns when i pee": "burning micturition", "uti": "burning micturition",
    "frequent urination": "bladder discomfort", "peeing a lot": "polyuria",
    "dark urine": "dark urine", "brown urine": "dark urine",
    "yellow eyes": "yellowing of eyes", "yellow urine": "yellow urine",

    # Sleep
    "cant sleep": "restlessness", "insomnia": "restlessness", "sleepless": "restlessness",
    "trouble sleeping": "restlessness",

    # Eyes
    "red eyes": "redness of eyes", "watery eyes": "watering from eyes",
    "teary eyes": "watering from eyes", "puffy eyes": "puffy face and eyes",
    "swollen eyes": "puffy face and eyes", "sunken eyes": "sunken eyes",

    # Swelling
    "swollen legs": "swollen legs", "leg swelling": "swollen legs",
    "swollen feet": "swollen legs", "swelling": "swelling joints",
    "puffy face": "puffy face and eyes", "swollen face": "puffy face and eyes",
    "swollen glands": "swelled lymph nodes", "swollen lymph nodes": "swelled lymph nodes",

    # Heart
    "fast heartbeat": "fast heart rate", "racing heart": "fast heart rate",
    "heart racing": "fast heart rate", "palpitations": "palpitations",
    "heart pounding": "palpitations",

    # Flu-like clusters (common casual phrases)
    "flu": "high fever", "influenza": "high fever",
    "cold": "continuous sneezing",
    "chesty": "phlegm", "chesty cough": "phlegm",

    # Other
    "dehydration": "dehydration", "thirsty": "excessive hunger",
    "increased thirst": "excessive hunger",
    "hair loss": "inflammatory nails", "losing hair": "inflammatory nails",
    "brittle nails": "brittle nails", "nail problems": "brittle nails",
    "cold hands": "cold hands and feets", "cold feet": "cold hands and feets",
    "tingling": "drying and tingling lips", "numbness": "muscle weakness",
    "pins and needles": "drying and tingling lips",
    "weak limbs": "weakness in limbs", "weak arms": "weakness in limbs",
    "weak legs": "weakness in limbs",
}


def extract_symptoms(user_text):
    user_text = user_text.lower()
    user_text = re.sub(r'[^\w\s]', ' ', user_text)

    detected = set()

    # 1. Multi-word synonym matching first (longer phrases take priority)
    sorted_synonyms = sorted(SYNONYM_MAP.keys(), key=len, reverse=True)
    matched_spans = []

    for key in sorted_synonyms:
        pattern = r'\b' + re.escape(key) + r'\b'
        if re.search(pattern, user_text):
            detected.add(SYNONYM_MAP[key])
            matched_spans.append(key)

    # 2. Direct match against symptoms_list
    for symptom in symptoms_list:
        clean = symptom.replace('_', ' ')
        if clean in user_text:
            detected.add(symptom)

    # 3. Fuzzy matching for unmatched words against symptoms_list
    user_words = user_text.split()
    symptom_names_clean = [s.replace('_', ' ') for s in symptoms_list]

    # Build bigrams and trigrams from user text for multi-word fuzzy matching
    ngrams = []
    for i in range(len(user_words)):
        ngrams.append(user_words[i])
        if i + 1 < len(user_words):
            ngrams.append(user_words[i] + ' ' + user_words[i + 1])
        if i + 2 < len(user_words):
            ngrams.append(user_words[i] + ' ' + user_words[i + 1] + ' ' + user_words[i + 2])

    for ng in ngrams:
        # Skip very short tokens
        if len(ng) < 4:
            continue
        # Skip if already matched by synonym
        if any(ng in span for span in matched_spans):
            continue
        matches = difflib.get_close_matches(ng, symptom_names_clean, n=1, cutoff=0.75)
        if matches:
            idx = symptom_names_clean.index(matches[0])
            detected.add(symptoms_list[idx])

    return list(detected)


# --- Groq LLM Context ---
async def fetch_groq_context(disease_name: str, symptoms: list) -> str:
    if not GROQ_API_KEY:
        return ""

    symptom_str = ", ".join([s.replace("_", " ") for s in symptoms])
    prompt = (
        f"A patient reports these symptoms: {symptom_str}. "
        f"The most likely condition is {disease_name}. "
        f"In 2-3 sentences, give an empathetic, plain-English explanation of what this condition is "
        f"and one practical next step the patient should take. "
        f"Do not use medical jargon. Do not give a diagnosis — frame it as educational information."
    )

    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.post(
                GROQ_URL,
                headers={
                    "Authorization": f"Bearer {GROQ_API_KEY}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": GROQ_MODEL,
                    "messages": [
                        {"role": "system", "content": "You are a compassionate medical information assistant. Keep responses brief and caring."},
                        {"role": "user", "content": prompt},
                    ],
                    "temperature": 0.6,
                    "max_tokens": 200,
                },
            )
            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"].strip()
    except Exception as e:
        print(f"Groq API error: {e}")
        return ""


@app.post("/predict")
async def predict(input_data: SymptomInput):
    user_symptoms = extract_symptoms(input_data.text)

    if not user_symptoms:
        return {
            "status": "failed",
            "message": "No symptoms detected. Try terms like 'fever', 'cough', 'rash', 'headache'."
        }

    # Vectorize
    input_vector = [1 if s in user_symptoms else 0 for s in symptoms_list]
    input_vector = np.array(input_vector).reshape(1, -1)

    # Raw Prediction
    probs = model.predict_proba(input_vector)[0]

    # Bayesian Re-ranking
    is_vague_input = len(user_symptoms) < 5

    adjusted_probs = []
    for i, prob in enumerate(probs):
        disease = le.inverse_transform([i])[0]
        severity = DISEASE_PREVALENCE.get(disease, 'MEDIUM')

        weight = 1.0
        if is_vague_input:
            if severity == 'HIGH':
                weight = 1.5
            elif severity == 'LOW':
                weight = 0.2

        # 1. Symptom co-occurrence bonus: Respiratory
        if disease in ['Pneumonia', 'Bronchial Asthma', 'Tuberculosis']:
            if 'high fever' in user_symptoms and any(s in user_symptoms for s in ['cough', 'breathlessness', 'mucoid sputum']):
                weight *= 1.8
        
        # Penalize Heart attack if respiratory combo matches
        if disease == 'Heart attack':
            if 'high fever' in user_symptoms and any(s in user_symptoms for s in ['cough', 'breathlessness', 'mucoid sputum']):
                weight *= 0.3
                
        # 2. Symptom co-occurrence bonus: Tropical/Fever
        if disease in ['Malaria', 'Dengue']:
            if 'high fever' in user_symptoms and any(s in user_symptoms for s in ['chills', 'sweating', 'vomiting']):
                weight *= 1.8
                
        # 3. Penalize Heart attack if lacking specific severe symptoms
        if disease == 'Heart attack':
            has_severe = any(s in user_symptoms for s in ['palpitations', 'sweating', 'vomiting', 'left arm pain']) or 'left arm' in input_data.text.lower()
            if not has_severe:
                weight *= 0.25

        new_score = prob * weight
        adjusted_probs.append((disease, new_score, severity))

    # Normalize the new scores so they sum up to 1.0
    total_score = sum(score for _, score, _ in adjusted_probs)
    if total_score > 0:
        adjusted_probs = [(disease, score / total_score, severity) for disease, score, severity in adjusted_probs]


    adjusted_probs.sort(key=lambda x: x[1], reverse=True)

    top_results = adjusted_probs[:3]

    results = []
    for i, (disease, score, severity) in enumerate(top_results):
        display_conf = min(score, 0.99)

        if display_conf > 0.01:
            ai_explanation = ""
            if i == 0:
                ai_explanation = await fetch_groq_context(disease, user_symptoms)

            results.append({
                "disease": disease,
                "confidence": round(display_conf * 100, 2),
                "severity": severity,
                "description": description_map.get(disease, "Definition unavailable."),
                "precautions": precaution_map.get(disease, []),
                "ai_explanation": ai_explanation,
            })

    return {
        "status": "success",
        "extracted_symptoms": list(user_symptoms),
        "predictions": results
    }