# 🏥 Healthcare Disease Prediction System

**Author:** Yug Pandya  
**Tech Stack:** Python · Scikit-learn · Flask · Pandas · NumPy

A machine learning-based web application that predicts diseases from symptoms using an ensemble of three ML models.

---

## 🎯 Features

- 🔬 **Ensemble ML** — Random Forest + Naive Bayes + SVM with majority voting
- 💊 **41 Diseases** predicted from **132 symptoms**
- 📊 **Top-5 probabilities** shown with confidence scores
- ⚠️ **Severity scoring** based on symptom weights
- 💡 **Precautions & descriptions** for every predicted disease
- 🌐 **Flask Web App** with symptom autocomplete UI
- 🔌 **REST API** ready for integration

---

## 🚀 How to Run

```bash
# 1. Clone the repo
git clone https://github.com/yug09-hub/Healthcare-Disease-Prediction
cd Healthcare-Disease-Prediction

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the web app
python app.py
```

Open your browser at **http://localhost:5000**

---

## 📁 Project Structure

```
Healthcare-Disease-Prediction/
├── disease_predictor.py   # Core ML engine (train + predict)
├── app.py                 # Flask web application + UI
├── requirements.txt       # Python dependencies
├── Training.csv           # Training dataset (4920 samples)
├── Testing.csv            # Testing dataset (41 samples)
└── README.md
```

---

## 🧠 ML Models

| Model | Test Accuracy |
|---|---|
| Random Forest | 100% |
| Naive Bayes | 100% |
| SVM (Linear) | 100% |
| **Ensemble (Voting)** | **100%** |

---

## 🔌 API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| POST | `/predict` | Predict disease from symptoms |
| GET | `/symptoms` | Get all 132 symptoms |
| GET | `/diseases` | Get all 41 disease classes |
| GET | `/health` | Check server status |

### Example API Request

```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"symptoms": ["itching", "skin_rash", "high_fever"]}'
```

### Example Response

```json
{
  "predicted_disease": "Fungal infection",
  "confidence": 100.0,
  "severity_score": 45.2,
  "all_predictions": {
    "RandomForest": "Fungal infection",
    "NaiveBayes": "Fungal infection",
    "SVM": "Fungal infection"
  },
  "probabilities": [
    { "disease": "Fungal infection", "score": 64.25 },
    { "disease": "Drug Reaction", "score": 2.61 }
  ],
  "disease_info": {
    "description": "A fungal infection caused by fungi that live on the skin.",
    "precautions": ["Keep skin clean and dry", "Use antifungal powder"]
  }
}
```

---

## 📦 Dependencies

```
pandas
numpy
scikit-learn
flask
```

---


## 📄 Base Research Paper

📖 https://ieeexplore.ieee.org/document/9154130

---

## 📬 Contact

**Email:** yug.work01@gmail.com  

