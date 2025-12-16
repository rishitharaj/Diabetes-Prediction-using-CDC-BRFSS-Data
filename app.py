"""
app.py

Flask app to serve the trained CatBoost diabetes risk model.

- GET  /        -> show Bootstrap form (form.html)
- POST /predict -> read form inputs, run model, show prediction on same page
"""

import os
import json
import joblib
import pandas as pd
from flask import Flask, render_template, request

# --- Paths ----------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "Diabetes CatBoost Model.pkl")
LABELS_PATH = os.path.join(BASE_DIR, "feature_labels.json")  # optional

# --- Flask app -----------------------------------------------
app = Flask(__name__)

# --- Load model artifact at startup ---------------------------
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"Model file not found at {MODEL_PATH}. "
        "Ensure Diabetes CatBoost Model.pkl is present."
    )

artifact = joblib.load(MODEL_PATH)

model = artifact["model"]
THRESHOLD = artifact["threshold"]
feature_names = artifact["features"]

# Friendly labels for form
if os.path.exists(LABELS_PATH):
    with open(LABELS_PATH, "r") as f:
        feature_labels = json.load(f)
else:
    feature_labels = {feat: feat for feat in feature_names}


# --- Routes ---------------------------------------------------

@app.route("/", methods=["GET"])
def index():
    """
    Show the form. On initial load, no prediction yet.
    """
    return render_template(
        "form.html",
        prediction_text=None,
        labels=feature_labels
    )


@app.route("/predict", methods=["POST"])
def predict():
    """
    Read values from the form, build a DataFrame in the correct
    column order, run the model, apply threshold, and show result.
    """
    try:
        values = []
        missing_fields = []

        for feat in feature_names:
            raw = request.form.get(feat)

            if raw is None or raw == "":
                missing_fields.append(feat)
                continue

            try:
                val = float(raw)
            except ValueError:
                low = raw.lower()
                if low in ["yes", "y", "true", "1"]:
                    val = 1.0
                else:
                    val = 0.0

            values.append(val)

        if missing_fields:
            return render_template(
                "form.html",
                labels=feature_labels,
                prediction_text=f"Missing or invalid values for: {', '.join(missing_fields)}"
            )

        # Build DataFrame in training column order
        X_df = pd.DataFrame([values], columns=feature_names)

        # Predict probability of diabetes (class 1)
        prob = model.predict_proba(X_df)[0, 1]

        # Apply optimized threshold
        pred = int(prob >= THRESHOLD)

        # Human-friendly message
        prob_text = f"{prob:.1%}"

        if pred == 1:
            prediction_text = (
                f"⚠️ High diabetes risk detected "
                f"(estimated probability: {prob_text}). "
                "This is a screening result, not a medical diagnosis."
            )
        else:
            prediction_text = (
                f"✅ Low diabetes risk detected "
                f"(estimated probability: {prob_text})."
            )

        return render_template(
            "form.html",
            labels=feature_labels,
            prediction_text=prediction_text
        )

    except Exception as e:
        return render_template(
            "form.html",
            labels=feature_labels,
            prediction_text=f"Error while predicting: {e}"
        )


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)

