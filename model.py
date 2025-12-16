# model.py
import json, os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
import warnings 
warnings.filterwarnings("ignore")


# === CONFIG ===
CSV_PATH = 'diabetes_binary_health_indicators_BRFSS2015.csv'
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "diabetes_catboost_model.pkl")
LABELS_PATH = os.path.join(BASE_DIR, "feature_labels.json")
TARGET = 'Diabetes_binary'


friendly_labels = {
    "HighBP": "Do you have high blood pressure?",
    "HighChol": "Do you have high cholesterol?",
    "CholCheck": "Have you checked your cholesterol?",
    "BMI": "What is your BMI (Body Mass Index)?",

    "Smoker": "Do you smoke?",
    "Stroke": "Did you have a stroke?",
    "HeartDiseaseorAttack": "Do you have heart disease or have you experienced a heart attack?",
    "PhysActivity": "Have you done any physical activity in the past 30 days?",

    "Fruits": "Do you consume fruits daily?",
    "Veggies": "Do you consume vegetables daily?",
    "HvyAlcoholConsump": "Do you have heavy alcohol consumption?",
    "AnyHealthcare": "Do you have any healthcare coverage?",
    "NoDocbcCost": "Have you avoided seeing a doctor because of cost?",

    "GenHlth": "How would you rate your general health?",
    "MentHlth": "In the past 30 days, how many days was your mental health not good?",
    "PhysHlth": "In the past 30 days, how many days was your physical health not good?",
    "DiffWalk": "Do you have serious difficulty walking or climbing stairs?",

    "Sex": "What is your sex?",
    "Age": "What is your age category?",
    "Education": "What is your highest level of education?",
    "Income": "What is your household’s annual income range?"
}

LABELS_PATH = os.path.join(BASE_DIR, "feature_labels.json")

# === Load model artifact ===
artifact = joblib.load(MODEL_PATH)

model = artifact["model"]
THRESHOLD = artifact["threshold"]
FEATURES = artifact["features"]

# === Load friendly labels ===
with open(LABELS_PATH, "r") as f:
    FEATURE_LABELS = json.load(f)


def preprocess_input(input_dict: dict) -> pd.DataFrame:
    """
    Convert user input dictionary into model-ready DataFrame
    """
    df = pd.DataFrame([input_dict])

    # Ensure correct feature order
    df = df[FEATURES]

    return df


def predict_diabetes(input_dict: dict) -> dict:
    """
    Returns probability and binary prediction using chosen threshold
    """
    X = preprocess_input(input_dict)

    prob = model.predict_proba(X)[0, 1]
    prediction = int(prob >= THRESHOLD)

    return {
        "probability": round(float(prob), 4),
        "prediction": prediction,
        "threshold": THRESHOLD
    }

if __name__ == "__main__":
    raise RuntimeError(
        "model.py is not executable. Run app.py instead."
    )
