"""
Disease Prediction System — Flask Web API
Author: Yug Pandya
GitHub: https://github.com/yug09-hub/Healthcare-Disease-Prediction

Run:
    pip install flask scikit-learn pandas numpy
    python app.py
Then open: http://localhost:5000
"""

from flask import Flask, request, jsonify, render_template_string
from disease_predictor import DiseasePredictionEngine
import os

app = Flask(__name__)

# ── Boot: train engine once on startup ───────────────────────────────────────
print("🚀 Booting Disease Prediction Engine...")
engine = DiseasePredictionEngine(train_path="Training.csv", test_path="Testing.csv")
engine.train(verbose=True)
print("✅ Engine ready.\n")


# ─────────────────────────────────────────────────────────────────────────────
# HTML UI (single-file, no extra templates needed)
# ─────────────────────────────────────────────────────────────────────────────

HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
  <title>Disease Prediction System</title>
  <style>
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body { font-family: 'Segoe UI', sans-serif; background: #f0f4f8; color: #2d3748; }

    header {
      background: linear-gradient(135deg, #2b6cb0, #4299e1);
      color: #fff; padding: 24px 40px;
      box-shadow: 0 2px 8px rgba(0,0,0,.2);
    }
    header h1 { font-size: 1.8rem; }
    header p  { font-size: .9rem; opacity: .85; margin-top: 4px; }

    .container { max-width: 900px; margin: 32px auto; padding: 0 16px; }

    .card {
      background: #fff; border-radius: 12px;
      box-shadow: 0 2px 12px rgba(0,0,0,.08);
      padding: 28px; margin-bottom: 24px;
    }
    .card h2 { font-size: 1.15rem; color: #2b6cb0; margin-bottom: 16px; }

    label { font-size: .85rem; font-weight: 600; display: block; margin-bottom: 6px; }
    input[type=text] {
      width: 100%; padding: 10px 14px; border: 1px solid #cbd5e0;
      border-radius: 8px; font-size: .95rem; outline: none;
      transition: border .2s;
    }
    input[type=text]:focus { border-color: #4299e1; }

    .symptom-chips { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 10px; min-height: 40px; }
    .chip {
      background: #ebf8ff; border: 1px solid #90cdf4;
      border-radius: 20px; padding: 4px 12px;
      font-size: .82rem; cursor: pointer; display: flex; align-items: center; gap: 6px;
    }
    .chip span { color: #e53e3e; font-weight: bold; }

    .suggestions {
      background: #fff; border: 1px solid #e2e8f0;
      border-radius: 8px; max-height: 180px; overflow-y: auto;
      position: absolute; z-index: 10; width: 100%;
    }
    .suggestions div {
      padding: 8px 14px; cursor: pointer; font-size: .9rem;
    }
    .suggestions div:hover { background: #ebf8ff; }
    .relative { position: relative; }

    button.predict-btn {
      margin-top: 18px; width: 100%; padding: 12px;
      background: linear-gradient(135deg, #2b6cb0, #4299e1);
      color: #fff; border: none; border-radius: 8px;
      font-size: 1rem; font-weight: 600; cursor: pointer;
      transition: opacity .2s;
    }
    button.predict-btn:hover  { opacity: .9; }
    button.predict-btn:active { opacity: .8; }

    .result-card { display: none; }
    .disease-name {
      font-size: 1.6rem; font-weight: 700; color: #2b6cb0;
      border-bottom: 2px solid #bee3f8; padding-bottom: 10px; margin-bottom: 14px;
    }
    .badges { display: flex; gap: 10px; flex-wrap: wrap; margin-bottom: 16px; }
    .badge {
      border-radius: 20px; padding: 4px 14px;
      font-size: .82rem; font-weight: 600;
    }
    .badge-green  { background: #c6f6d5; color: #276749; }
    .badge-yellow { background: #fefcbf; color: #744210; }
    .badge-red    { background: #fed7d7; color: #9b2c2c; }

    .description { font-size: .93rem; color: #4a5568; margin-bottom: 16px; line-height: 1.6; }

    .section-title { font-weight: 600; color: #2d3748; margin-bottom: 8px; }
    .precautions { list-style: none; }
    .precautions li { padding: 6px 0; font-size: .9rem; display: flex; align-items: center; gap: 8px; }
    .precautions li::before { content: "✔"; color: #38a169; font-weight: bold; }

    .model-votes { display: flex; gap: 12px; flex-wrap: wrap; margin-bottom: 16px; }
    .vote-box {
      background: #f7fafc; border: 1px solid #e2e8f0;
      border-radius: 8px; padding: 8px 14px; font-size: .82rem;
    }
    .vote-box strong { display: block; color: #4299e1; }

    .top5 { margin-top: 16px; }
    .bar-row { display: flex; align-items: center; gap: 10px; margin-bottom: 6px; }
    .bar-label { width: 200px; font-size: .82rem; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
    .bar-track { flex: 1; background: #e2e8f0; border-radius: 4px; height: 10px; }
    .bar-fill  { background: #4299e1; height: 10px; border-radius: 4px; }
    .bar-pct   { font-size: .78rem; color: #718096; width: 40px; text-align: right; }

    footer { text-align: center; font-size: .8rem; color: #a0aec0; margin: 32px 0 16px; }
  </style>
</head>
<body>

<header>
  <h1>🏥 Healthcare Disease Prediction System</h1>
  <p>Ensemble ML model (Random Forest + Naive Bayes + SVM) trained on 4,920 samples · 41 diseases · 132 symptoms</p>
</header>

<div class="container">

  <!-- Input Card -->
  <div class="card">
    <h2>🩺 Enter Your Symptoms</h2>
    <label for="symptom-input">Type a symptom and select from suggestions:</label>
    <div class="relative">
      <input type="text" id="symptom-input" placeholder="e.g. itching, skin rash, fever..." autocomplete="off" />
      <div class="suggestions" id="suggestions" style="display:none"></div>
    </div>
    <div class="symptom-chips" id="chips"></div>
    <button class="predict-btn" onclick="predict()">🔍 Predict Disease</button>
  </div>

  <!-- Result Card -->
  <div class="card result-card" id="result-card">
    <div class="disease-name" id="disease-name"></div>
    <div class="badges" id="badges"></div>
    <p class="description" id="description"></p>

    <p class="section-title">Precautions:</p>
    <ul class="precautions" id="precautions"></ul>

    <p class="section-title" style="margin-top:16px">Model Votes:</p>
    <div class="model-votes" id="model-votes"></div>

    <div class="top5">
      <p class="section-title">Top 5 Possibilities:</p>
      <div id="top5-bars"></div>
    </div>
  </div>

</div>

<footer>Disease Prediction System &nbsp;·&nbsp; Yug Pandya &nbsp;·&nbsp;
  <a href="https://github.com/yug09-hub/Healthcare-Disease-Prediction" target="_blank">GitHub</a>
</footer>

<script>
  const ALL_SYMPTOMS = {{ symptoms|tojson }};
  let selected = [];

  const input   = document.getElementById("symptom-input");
  const sugg    = document.getElementById("suggestions");
  const chips   = document.getElementById("chips");

  input.addEventListener("input", () => {
    const q = input.value.trim().toLowerCase().replace(/ /g,"_");
    if (!q) { sugg.style.display="none"; return; }
    const matches = ALL_SYMPTOMS.filter(s => s.includes(q) && !selected.includes(s)).slice(0,10);
    if (!matches.length) { sugg.style.display="none"; return; }
    sugg.innerHTML = matches.map(s =>
      `<div onclick="addSymptom('${s}')">${s.replace(/_/g," ")}</div>`
    ).join("");
    sugg.style.display = "block";
  });

  document.addEventListener("click", e => {
    if (!sugg.contains(e.target) && e.target !== input) sugg.style.display="none";
  });

  function addSymptom(s) {
    if (selected.includes(s)) return;
    selected.push(s);
    sugg.style.display = "none";
    input.value = "";
    renderChips();
  }

  function removeSymptom(s) {
    selected = selected.filter(x => x !== s);
    renderChips();
  }

  function renderChips() {
    chips.innerHTML = selected.map(s =>
      `<div class="chip">${s.replace(/_/g," ")} <span onclick="removeSymptom('${s}')">✕</span></div>`
    ).join("");
  }

  async function predict() {
    if (!selected.length) { alert("Please add at least one symptom."); return; }
    const res  = await fetch("/predict", {
      method: "POST",
      headers: {"Content-Type":"application/json"},
      body: JSON.stringify({symptoms: selected})
    });
    const data = await res.json();
    showResult(data);
  }

  function severityBadge(score) {
    if (score < 35) return ["badge-green",  "Low Severity"];
    if (score < 65) return ["badge-yellow", "Moderate Severity"];
    return                  ["badge-red",   "High Severity"];
  }

  function showResult(r) {
    document.getElementById("disease-name").textContent = r.predicted_disease;
    document.getElementById("description").textContent  = r.disease_info.description;

    const [sc, sl] = severityBadge(r.severity_score);
    document.getElementById("badges").innerHTML =
      `<span class="badge badge-green">Confidence: ${r.confidence}%</span>
       <span class="badge ${sc}">${sl} (${r.severity_score}%)</span>`;

    document.getElementById("precautions").innerHTML =
      r.disease_info.precautions.map(p => `<li>${p}</li>`).join("");

    document.getElementById("model-votes").innerHTML =
      Object.entries(r.all_predictions).map(([model, disease]) =>
        `<div class="vote-box"><strong>${model}</strong>${disease}</div>`
      ).join("");

    const maxScore = r.probabilities[0].score || 1;
    document.getElementById("top5-bars").innerHTML = r.probabilities.map(p => `
      <div class="bar-row">
        <div class="bar-label" title="${p.disease}">${p.disease}</div>
        <div class="bar-track"><div class="bar-fill" style="width:${(p.score/maxScore*100).toFixed(1)}%"></div></div>
        <div class="bar-pct">${p.score}%</div>
      </div>`
    ).join("");

    const card = document.getElementById("result-card");
    card.style.display = "block";
    card.scrollIntoView({ behavior: "smooth" });
  }
</script>

</body>
</html>
"""


# ─────────────────────────────────────────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    symptoms = engine.get_all_symptoms()
    return render_template_string(HTML, symptoms=symptoms)


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(force=True)
    symptoms = data.get("symptoms", [])

    if not symptoms:
        return jsonify({"error": "No symptoms provided"}), 400

    try:
        result = engine.predict(symptoms)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/symptoms", methods=["GET"])
def get_symptoms():
    """Returns the full list of recognized symptoms."""
    return jsonify(engine.get_all_symptoms())


@app.route("/diseases", methods=["GET"])
def get_diseases():
    """Returns all known disease classes."""
    return jsonify(engine.classes)


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "model_trained": engine.is_trained})


# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
