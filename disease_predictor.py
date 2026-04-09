"""
Disease Prediction System - Core Engine
Author: Yug Pandya
GitHub: https://github.com/yug09-hub/Healthcare-Disease-Prediction

Uses an ensemble of Random Forest, Naive Bayes, and SVM classifiers
trained on symptom data to predict diseases.
"""

import pandas as pd
import numpy as np
import pickle
import os
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from collections import Counter

# ─────────────────────────────────────────────────────────────────────────────
# DISEASE DESCRIPTIONS & PRECAUTIONS
# ─────────────────────────────────────────────────────────────────────────────

DISEASE_INFO = {
    "Fungal infection": {
        "description": "A fungal infection caused by fungi that live on the skin, hair, or nails.",
        "precautions": ["Keep skin clean and dry", "Avoid sharing personal items", "Wear breathable clothing", "Use antifungal powder"]
    },
    "Allergy": {
        "description": "An immune system reaction to a foreign substance not typically harmful to your body.",
        "precautions": ["Avoid known allergens", "Keep windows closed during high pollen", "Use air purifiers", "Carry antihistamines"]
    },
    "GERD": {
        "description": "Gastroesophageal reflux disease — acid from the stomach flows back into the esophagus.",
        "precautions": ["Avoid spicy food", "Don't lie down after eating", "Elevate head while sleeping", "Eat smaller meals"]
    },
    "Chronic cholestasis": {
        "description": "A condition where bile flow from the liver is reduced or blocked.",
        "precautions": ["Consult a doctor immediately", "Avoid alcohol", "Follow a low-fat diet", "Take prescribed medications"]
    },
    "Drug Reaction": {
        "description": "An adverse reaction to a medication, ranging from mild to severe.",
        "precautions": ["Stop the suspected drug", "Consult your doctor", "Carry a list of drug allergies", "Take antihistamines if mild"]
    },
    "Peptic ulcer diseae": {
        "description": "Sores that develop on the lining of the stomach, small intestine, or esophagus.",
        "precautions": ["Avoid spicy food and alcohol", "Stop smoking", "Take antacids", "Reduce stress"]
    },
    "AIDS": {
        "description": "Acquired Immunodeficiency Syndrome — a chronic condition caused by HIV.",
        "precautions": ["Use protection during intercourse", "Don't share needles", "Take antiretroviral therapy", "Regular medical checkups"]
    },
    "Diabetes ": {
        "description": "A chronic disease that occurs when the body cannot properly regulate blood sugar levels.",
        "precautions": ["Monitor blood sugar regularly", "Exercise daily", "Follow a balanced diet", "Take insulin or prescribed medication"]
    },
    "Gastroenteritis": {
        "description": "Inflammation of the stomach and intestines, typically from a viral or bacterial infection.",
        "precautions": ["Stay hydrated with ORS", "Avoid dairy products temporarily", "Eat bland foods", "Wash hands frequently"]
    },
    "Bronchial Asthma": {
        "description": "A condition in which the airways narrow and swell, causing breathing difficulty.",
        "precautions": ["Use inhalers as prescribed", "Avoid smoke and dust", "Monitor peak flow", "Keep rescue inhaler handy"]
    },
    "Hypertension ": {
        "description": "High blood pressure — a condition where the long-term force of blood against artery walls is too high.",
        "precautions": ["Reduce salt intake", "Exercise regularly", "Avoid stress", "Take blood pressure medications"]
    },
    "Migraine": {
        "description": "A neurological condition causing intense, throbbing headaches often with nausea and light sensitivity.",
        "precautions": ["Rest in a dark quiet room", "Apply cold/warm compress", "Avoid triggers like caffeine", "Take prescribed pain relief"]
    },
    "Cervical spondylosis": {
        "description": "Age-related wear affecting spinal disks in the neck, causing neck pain and stiffness.",
        "precautions": ["Do neck exercises", "Use ergonomic pillow", "Avoid long screen time", "Apply heat/ice to neck"]
    },
    "Paralysis (brain hemorrhage)": {
        "description": "Loss of muscle function due to bleeding in or around the brain.",
        "precautions": ["Seek emergency care immediately", "Follow physiotherapy", "Control blood pressure", "Avoid strenuous activity"]
    },
    "Jaundice": {
        "description": "Yellowing of the skin and eyes caused by high bilirubin levels in the blood.",
        "precautions": ["Drink plenty of water", "Avoid alcohol completely", "Rest and eat light foods", "Consult a liver specialist"]
    },
    "Malaria": {
        "description": "A mosquito-borne disease caused by Plasmodium parasites, causing fever and chills.",
        "precautions": ["Use mosquito nets", "Apply mosquito repellent", "Take antimalarial drugs", "Wear long sleeves outdoors"]
    },
    "Chicken pox": {
        "description": "A highly contagious viral infection causing an itchy blister-like rash.",
        "precautions": ["Stay isolated from others", "Avoid scratching blisters", "Apply calamine lotion", "Take antihistamines for itch"]
    },
    "Dengue": {
        "description": "A mosquito-borne viral infection causing fever, headache, and severe joint pain.",
        "precautions": ["Use mosquito repellent", "Eliminate standing water", "Stay hydrated", "Monitor platelet count"]
    },
    "Typhoid": {
        "description": "A bacterial infection caused by Salmonella typhi, spread through contaminated food and water.",
        "precautions": ["Drink boiled/purified water", "Eat hygienic food", "Complete antibiotic course", "Get vaccinated"]
    },
    "Hepatitis A": {
        "description": "A highly contagious liver infection caused by the hepatitis A virus.",
        "precautions": ["Get vaccinated", "Wash hands thoroughly", "Avoid contaminated food/water", "Rest and stay hydrated"]
    },
    "Hepatitis B": {
        "description": "A serious liver infection caused by the hepatitis B virus (HBV).",
        "precautions": ["Get vaccinated", "Use protection during intercourse", "Don't share needles", "Regular liver monitoring"]
    },
    "Hepatitis C": {
        "description": "A viral infection that causes liver inflammation, sometimes leading to serious liver damage.",
        "precautions": ["Don't share needles or razors", "Use protection during intercourse", "Get tested regularly", "Follow antiviral therapy"]
    },
    "Hepatitis D": {
        "description": "A serious liver disease caused by HDV, which only affects people with hepatitis B.",
        "precautions": ["Get Hepatitis B vaccine", "Avoid sharing needles", "Use protection during intercourse", "Regular medical follow-up"]
    },
    "Hepatitis E": {
        "description": "A liver disease caused by HEV, primarily spread through contaminated water.",
        "precautions": ["Drink purified water", "Eat well-cooked food", "Practice good hygiene", "Avoid travel to high-risk areas"]
    },
    "Alcoholic hepatitis": {
        "description": "Inflammation of the liver caused by drinking too much alcohol.",
        "precautions": ["Stop alcohol consumption immediately", "Consult a gastroenterologist", "Eat nutritious food", "Take prescribed steroids"]
    },
    "Tuberculosis": {
        "description": "A serious infectious disease caused by Mycobacterium tuberculosis, mainly affecting the lungs.",
        "precautions": ["Complete the full TB medication course", "Cover mouth while coughing", "Maintain good ventilation", "Regular follow-up tests"]
    },
    "Common Cold": {
        "description": "A viral infection of your nose and throat causing runny nose, cough, and sore throat.",
        "precautions": ["Stay hydrated", "Rest well", "Use steam inhalation", "Avoid cold food and drinks"]
    },
    "Pneumonia": {
        "description": "An infection that inflames air sacs in one or both lungs, which may fill with fluid.",
        "precautions": ["Complete antibiotic course", "Rest and stay hydrated", "Use humidifier", "Seek medical care promptly"]
    },
    "Dimorphic hemmorhoids(piles)": {
        "description": "Swollen veins in the rectum or anus that cause discomfort and bleeding.",
        "precautions": ["Eat high-fiber diet", "Stay hydrated", "Avoid straining during bowel movements", "Take sitz baths"]
    },
    "Heart attack": {
        "description": "A blockage of blood flow to the heart muscle, a medical emergency.",
        "precautions": ["Call emergency services immediately", "Chew aspirin if not allergic", "Rest and stay calm", "Avoid exertion"]
    },
    "Varicose veins": {
        "description": "Enlarged, twisted veins most commonly appearing in the legs.",
        "precautions": ["Elevate legs when resting", "Wear compression stockings", "Exercise regularly", "Avoid prolonged standing"]
    },
    "Hypothyroidism": {
        "description": "A condition where the thyroid gland doesn't produce enough thyroid hormone.",
        "precautions": ["Take levothyroxine as prescribed", "Monitor thyroid levels regularly", "Eat iodine-rich foods", "Avoid excessive soy intake"]
    },
    "Hyperthyroidism": {
        "description": "A condition where the thyroid gland produces too much thyroid hormone.",
        "precautions": ["Take antithyroid medications", "Avoid iodine-rich foods", "Reduce stress", "Monitor heart rate"]
    },
    "Hypoglycemia": {
        "description": "Abnormally low blood sugar levels causing shakiness, confusion, and sweating.",
        "precautions": ["Eat regular meals", "Carry glucose tablets", "Monitor blood sugar", "Avoid skipping meals"]
    },
    "Osteoarthristis": {
        "description": "Degeneration of joint cartilage causing pain and stiffness, especially in the knees and hips.",
        "precautions": ["Exercise regularly", "Maintain healthy weight", "Use joint support braces", "Take prescribed pain relief"]
    },
    "Arthritis": {
        "description": "Inflammation of joints causing pain and stiffness that worsens with age.",
        "precautions": ["Stay active with gentle exercises", "Apply hot/cold packs", "Take prescribed anti-inflammatories", "Maintain healthy weight"]
    },
    "(vertigo) Paroymsal  Positional Vertigo": {
        "description": "A condition causing brief episodes of dizziness triggered by head position changes.",
        "precautions": ["Perform Epley maneuver", "Avoid sudden head movements", "Sleep with head elevated", "Consult an ENT specialist"]
    },
    "Acne": {
        "description": "A skin condition that occurs when hair follicles are plugged with oil and dead skin cells.",
        "precautions": ["Wash face twice daily", "Avoid touching your face", "Use non-comedogenic products", "Don't pop pimples"]
    },
    "Urinary tract infection": {
        "description": "An infection in any part of the urinary system — kidneys, bladder, or urethra.",
        "precautions": ["Drink plenty of water", "Urinate frequently", "Wipe front to back", "Avoid holding urine for long"]
    },
    "Psoriasis": {
        "description": "A skin disease causing a rash with itchy, scaly patches, most commonly on knees and elbows.",
        "precautions": ["Moisturize skin daily", "Avoid triggers like stress", "Use prescribed topical treatments", "Get moderate sun exposure"]
    },
    "Impetigo": {
        "description": "A highly contagious bacterial skin infection causing red sores and honey-colored crusts.",
        "precautions": ["Apply antibiotic ointment", "Keep sores covered", "Wash hands frequently", "Avoid contact with others"]
    },
    "hepatitis A": {
        "description": "A highly contagious liver infection caused by the hepatitis A virus.",
        "precautions": ["Get vaccinated", "Wash hands thoroughly", "Avoid contaminated food/water", "Rest and stay hydrated"]
    },
}

DEFAULT_INFO = {
    "description": "A medical condition requiring professional diagnosis and treatment.",
    "precautions": ["Consult a doctor immediately", "Rest well", "Stay hydrated", "Avoid self-medication"]
}

# ─────────────────────────────────────────────────────────────────────────────
# SYMPTOM SEVERITY (0-7 scale for weighted prediction)
# ─────────────────────────────────────────────────────────────────────────────

SYMPTOM_SEVERITY = {
    "itching": 1, "skin_rash": 3, "nodal_skin_eruptions": 4, "continuous_sneezing": 4,
    "shivering": 5, "chills": 5, "joint_pain": 3, "stomach_pain": 5, "acidity": 3,
    "ulcers_on_tongue": 4, "muscle_wasting": 3, "vomiting": 5, "burning_micturition": 6,
    "fatigue": 4, "weight_gain": 3, "anxiety": 4, "mood_swings": 3, "weight_loss": 3,
    "restlessness": 5, "lethargy": 2, "cough": 4, "high_fever": 7, "breathlessness": 4,
    "sweating": 3, "dehydration": 4, "indigestion": 5, "headache": 3, "yellowish_skin": 3,
    "dark_urine": 4, "nausea": 5, "loss_of_appetite": 4, "pain_behind_the_eyes": 4,
    "back_pain": 3, "constipation": 4, "abdominal_pain": 5, "diarrhoea": 6, "mild_fever": 1,
    "yellow_urine": 1, "yellowing_of_eyes": 4, "acute_liver_failure": 7, "swelling_of_stomach": 7,
    "swelled_lymph_nodes": 6, "malaise": 1, "blurred_and_distorted_vision": 3, "phlegm": 2,
    "throat_irritation": 3, "redness_of_eyes": 4, "sinus_pressure": 4, "runny_nose": 5,
    "congestion": 4, "chest_pain": 7, "weakness_in_limbs": 7, "fast_heart_rate": 5,
    "neck_pain": 5, "dizziness": 4, "cramps": 4, "obesity": 4, "swollen_legs": 5,
    "enlarged_thyroid": 6, "excessive_hunger": 4, "slurred_speech": 4, "knee_pain": 3,
    "hip_joint_pain": 7, "muscle_weakness": 3, "stiff_neck": 5, "loss_of_balance": 4,
    "unsteadiness": 3, "loss_of_smell": 3, "bladder_discomfort": 4, "depression": 3,
    "irritability": 2, "muscle_pain": 3, "red_spots_over_body": 5, "belly_pain": 4,
    "palpitations": 4, "painful_walking": 4, "blackheads": 3, "skin_peeling": 3,
    "blister": 4, "coma": 7, "stomach_bleeding": 7, "blood_in_sputum": 5,
    "polyuria": 4, "family_history": 5, "receiving_blood_transfusion": 6,
    "receiving_unsterile_injections": 6, "history_of_alcohol_consumption": 5,
}


# ─────────────────────────────────────────────────────────────────────────────
# CORE ENGINE CLASS
# ─────────────────────────────────────────────────────────────────────────────

class DiseasePredictionEngine:
    """
    Ensemble Disease Prediction Engine.
    Trains Random Forest, Naive Bayes, and SVM classifiers and uses
    majority voting for final prediction.
    """

    def __init__(self, train_path="Training.csv", test_path="Testing.csv"):
        self.train_path = train_path
        self.test_path = test_path
        self.label_encoder = LabelEncoder()
        self.models = {}
        self.symptoms_list = []
        self.classes = []
        self.is_trained = False

    # ── Data Loading ──────────────────────────────────────────────────────────

    def load_data(self):
        """Load and prepare training and testing data."""
        train_df = pd.read_csv(self.train_path)
        test_df  = pd.read_csv(self.test_path)

        # Drop duplicate fluid_overload column if present
        train_df = train_df.loc[:, ~train_df.columns.duplicated()]
        test_df  = test_df.loc[:,  ~test_df.columns.duplicated()]

        # Separate features and target
        self.symptoms_list = [c for c in train_df.columns if c != "prognosis"]

        X_train = train_df[self.symptoms_list].fillna(0).astype(int)
        y_train = self.label_encoder.fit_transform(train_df["prognosis"])

        X_test  = test_df[self.symptoms_list].fillna(0).astype(int)
        y_test  = self.label_encoder.transform(test_df["prognosis"])

        self.classes = list(self.label_encoder.classes_)
        return X_train, y_train, X_test, y_test

    # ── Training ──────────────────────────────────────────────────────────────

    def train(self, verbose=True):
        """Train Random Forest, Naive Bayes, and SVM classifiers."""
        X_train, y_train, X_test, y_test = self.load_data()

        model_configs = {
            "RandomForest": RandomForestClassifier(
                n_estimators=200, max_depth=15, random_state=42, n_jobs=-1
            ),
            "NaiveBayes": GaussianNB(),
            "SVM": SVC(kernel="linear", probability=True, random_state=42),
        }

        results = {}
        for name, model in model_configs.items():
            if verbose:
                print(f"  Training {name}...", end=" ")

            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            acc   = accuracy_score(y_test, preds)
            self.models[name] = model
            results[name] = acc

            if verbose:
                print(f"Test Accuracy: {acc * 100:.2f}%")

        self.is_trained = True

        if verbose:
            print(f"\n✅ Training complete. Models: {list(self.models.keys())}")

        return results

    # ── Prediction ────────────────────────────────────────────────────────────

    def _symptoms_to_vector(self, symptoms_input):
        """
        Convert a list of symptom strings into a binary feature vector.
        Handles fuzzy matching for minor spelling differences.
        """
        vector = np.zeros(len(self.symptoms_list), dtype=int)
        unrecognized = []

        for symptom in symptoms_input:
            symptom_clean = symptom.strip().lower().replace(" ", "_")

            # Exact match
            if symptom_clean in self.symptoms_list:
                idx = self.symptoms_list.index(symptom_clean)
                vector[idx] = 1
                continue

            # Partial / fuzzy match
            matched = False
            for i, s in enumerate(self.symptoms_list):
                if symptom_clean in s or s in symptom_clean:
                    vector[i] = 1
                    matched = True
                    break

            if not matched:
                unrecognized.append(symptom)

        return vector, unrecognized

    def predict(self, symptoms_input):
        """
        Predict disease from a list of symptom strings.

        Args:
            symptoms_input (list[str]): e.g. ["itching", "skin_rash", "high_fever"]

        Returns:
            dict with keys:
                predicted_disease  – final ensemble prediction
                confidence         – confidence % of top model
                all_predictions    – each model's prediction
                probabilities      – top-5 diseases with scores
                disease_info       – description + precautions
                unrecognized       – symptoms not matched
                severity_score     – computed severity of entered symptoms
        """
        if not self.is_trained:
            raise RuntimeError("Model not trained. Call engine.train() first.")

        vector, unrecognized = self._symptoms_to_vector(symptoms_input)
        X = pd.DataFrame([vector], columns=self.symptoms_list)

        votes = []
        all_predictions = {}
        proba_sum = np.zeros(len(self.classes))

        for name, model in self.models.items():
            pred_idx  = model.predict(X)[0]
            pred_name = self.label_encoder.inverse_transform([pred_idx])[0]
            all_predictions[name] = pred_name
            votes.append(pred_name)

            if hasattr(model, "predict_proba"):
                proba_sum += model.predict_proba(X)[0]

        # Majority vote
        vote_counter   = Counter(votes)
        final_disease  = vote_counter.most_common(1)[0][0]

        # Average probability across models
        avg_proba = proba_sum / len(self.models)
        top5_idx  = np.argsort(avg_proba)[::-1][:5]
        top5      = [
            {"disease": self.classes[i], "score": round(float(avg_proba[i]) * 100, 2)}
            for i in top5_idx
        ]

        # Confidence: how many models agreed
        confidence = round(vote_counter[final_disease] / len(self.models) * 100, 1)

        # Severity score
        severity = sum(SYMPTOM_SEVERITY.get(s.strip().lower().replace(" ", "_"), 2)
                       for s in symptoms_input)
        max_possible = 7 * len(symptoms_input) if symptoms_input else 1
        severity_pct = round(severity / max_possible * 100, 1) if max_possible else 0

        info = DISEASE_INFO.get(final_disease, DEFAULT_INFO)

        return {
            "predicted_disease": final_disease,
            "confidence":        confidence,
            "all_predictions":   all_predictions,
            "probabilities":     top5,
            "disease_info":      info,
            "unrecognized":      unrecognized,
            "severity_score":    severity_pct,
        }

    # ── Evaluation ────────────────────────────────────────────────────────────

    def evaluate(self):
        """Print detailed evaluation report for all models."""
        X_train, y_train, X_test, y_test = self.load_data()

        for name, model in self.models.items():
            print(f"\n{'='*50}")
            print(f"  {name} — Evaluation Report")
            print(f"{'='*50}")
            preds = model.predict(X_test)
            print(classification_report(
                y_test, preds,
                target_names=self.label_encoder.classes_,
                zero_division=0
            ))

    def get_all_symptoms(self):
        """Return the full list of recognized symptoms."""
        return sorted(self.symptoms_list)

    # ── Model Persistence ─────────────────────────────────────────────────────

    def save(self, path="disease_model.pkl"):
        """Save trained engine to disk."""
        import pickle
        with open(path, "wb") as f:
            pickle.dump(self, f)
        print(f"✅ Model saved to {path}")

    @staticmethod
    def load(path="disease_model.pkl"):
        """Load a previously saved engine from disk."""
        import pickle
        with open(path, "rb") as f:
            engine = pickle.load(f)
        print(f"✅ Model loaded from {path}")
        return engine


# ─────────────────────────────────────────────────────────────────────────────
# CLI — Quick Demo
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 55)
    print("  🏥  Disease Prediction System — Core Engine")
    print("=" * 55)

    engine = DiseasePredictionEngine(
        train_path="Training.csv",
        test_path="Testing.csv"
    )

    print("\n📊 Training ensemble models...\n")
    engine.train(verbose=True)

    # ── Demo Predictions ──────────────────────────────────────────────────────
    test_cases = [
        ["itching", "skin_rash", "nodal_skin_eruptions"],
        ["high_fever", "headache", "nausea", "vomiting", "fatigue"],
        ["chest_pain", "fast_heart_rate", "breathlessness", "sweating"],
        ["yellowish_skin", "dark_urine", "nausea", "loss_of_appetite", "fatigue"],
    ]

    print("\n" + "=" * 55)
    print("  🔬  Demo Predictions")
    print("=" * 55)

    for symptoms in test_cases:
        result = engine.predict(symptoms)
        print(f"\n🩺 Symptoms   : {', '.join(symptoms)}")
        print(f"   Disease    : {result['predicted_disease']}")
        print(f"   Confidence : {result['confidence']}%  |  Severity: {result['severity_score']}%")
        print(f"   Models     : {result['all_predictions']}")
        top3 = [p['disease'] + " (" + str(p['score']) + "%)" for p in result['probabilities'][:3]]
        print(f"   Top-3      : {top3}")
        print(f"   Precaution : {result['disease_info']['precautions'][0]}")

    engine.save("disease_model.pkl")
    print("\n✅ Done! Use DiseasePredictionEngine in your app by importing disease_predictor.py")
