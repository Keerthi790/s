import streamlit as st
import pandas as pd
import numpy as np
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# Dataset file names
DATASETS = {
    "diabetes": ["diabetes.csv"],
    "heart":    ["heart.csv"],
    "kidney":   ["kidney_disease.csv", "kidney.csv"],
    "liver":    ["Indian Liver Patient Dataset (ILPD).csv", "liver.csv"]
}

# Manually defined normal ranges
NORMAL_RANGES = {
    "diabetes": {
        "Pregnancies": "0–6",
        "Glucose": "70–140 mg/dL",
        "BloodPressure": "60–90 mmHg",
        "SkinThickness": "10–50 mm",
        "Insulin": "15–276 mu U/ml",
        "BMI": "18.5–24.9",
        "DiabetesPedigreeFunction": "0.1–1.0",
        "Age": "20–60"
    },
    "heart": {
        "age": "29–77 years",
        "sex": "0 = Female, 1 = Male",
        "cp": "0–3 (Chest Pain Type)",
        "trestbps": "90–140 mmHg",
        "chol": "125–250 mg/dL",
        "fbs": "0 = False, 1 = True (Fasting Blood Sugar > 120 mg/dL)",
        "restecg": "0–2 (ECG Results)",
        "thalach": "100–200 bpm",
        "exang": "0 = No, 1 = Yes",
        "oldpeak": "0–4",
        "slope": "0–2",
        "ca": "0–4 (number of major vessels)",
        "thal": "1–3"
    },
    "kidney": {
        "age": "20–80 years",
        "bp": "60–120 mmHg",
        "sg": "1.005–1.025",
        "al": "0–4",
        "su": "0–5",
        "bgr": "70–200 mg/dL",
        "bu": "10–50 mg/dL",
        "sc": "0.4–1.5 mg/dL",
        "sod": "135–145 mEq/L",
        "pot": "3.5–5.5 mEq/L",
        "hemo": "12–16 g/dL",
        "pcv": "35–50%",
        "wc": "4000–11000 cells/cmm",
        "rc": "4.5–5.5 million/cmm"
    },
    "liver": {
        "Age": "20–80 years",
        "Gender": "0 = Female, 1 = Male",
        "Total_Bilirubin": "0.1–1.2 mg/dL",
        "Direct_Bilirubin": "0–0.3 mg/dL",
        "Alkaline_Phosphotase": "30–120 IU/L",
        "Alamine_Aminotransferase": "7–56 IU/L",
        "Aspartate_Aminotransferase": "10–40 IU/L",
        "Total_Proteins": "6–8.3 g/dL",
        "Albumin": "3.5–5.5 g/dL",
        "Albumin_and_Globulin_Ratio": "1.1–2.5"
    }
}

models = {}
scalers = {}

# Helper functions
def locate_file(candidates):
    for fn in candidates:
        if os.path.exists(fn):
            return fn
    raise FileNotFoundError(f"None of these files found: {candidates}")

def preprocess_data(name, df):
    df = df.copy()
    df.replace("?", np.nan, inplace=True)
    df.dropna(inplace=True)

    if name == "kidney":
        for col in df.select_dtypes(include='object').columns:
            df[col] = df[col].str.lower()
        if 'classification' in df.columns:
            df['classification'] = df['classification'].map({'ckd':1, 'notckd':0})
        df = df.select_dtypes(include=[np.number])

    elif name == "liver":
        if 'Gender' in df.columns:
            df['Gender'] = df['Gender'].map({'Male':1, 'Female':0})
        if 'Dataset' in df.columns:
            df['Dataset'] = df['Dataset'].replace({1:1, 2:0})

    return df

def get_target_column(name, df):
    cands = {
        "diabetes": [df.columns[-1]],
        "heart":    [df.columns[-1]],
        "kidney":   ["classification", "class", "target"],
        "liver":    ["Dataset", "class", "target"]
    }
    for col in cands[name]:
        if col in df.columns:
            return col
    return df.columns[-1]

def train_one(name):
    path = locate_file(DATASETS[name])
    df = pd.read_csv(path)
    df = preprocess_data(name, df)

    target = get_target_column(name, df)
    X = df.drop(columns=[target])
    y = df[target]

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    Xtr, Xte, ytr, yte = train_test_split(Xs, y, test_size=0.2, random_state=42)

    m = RandomForestClassifier(n_estimators=100, random_state=42)
    m.fit(Xtr, ytr)
    acc = accuracy_score(yte, m.predict(Xte))
    print(f"✅ {name.capitalize()} model accuracy: {acc:.2f}")

    models[name] = m
    scalers[name] = scaler

def load_and_train_all():
    for name in DATASETS:
        train_one(name)

def predict_disease(name, inputs):
    if name not in models:
        raise ValueError(f"Model for '{name}' not loaded")
    m = models[name]
    s = scalers[name]
    return m.predict(s.transform([inputs]))[0]

def get_feature_info(name):
    path = locate_file(DATASETS[name])
    df = pd.read_csv(path)
    df = preprocess_data(name, df)
    target = get_target_column(name, df)
    X = df.drop(columns=[target])
    return list(X.columns)

# === Streamlit App ===

st.set_page_config(page_title="Multi-Disease Prediction", layout="centered")
st.title("🧠 Multi-Disease Prediction System")

# Load models
load_and_train_all()

# Disease selector
selected_disease = st.sidebar.radio(
    "Select Disease",
    options=["diabetes", "heart", "kidney", "liver"],
    format_func=lambda x: x.capitalize() + " Disease Prediction"
)

st.subheader(f"📝 Enter Input for {selected_disease.capitalize()} Disease Prediction")

# Get input fields
features = get_feature_info(selected_disease)
user_inputs = []

for feature in features:
    val = st.number_input(label=feature, value=0.0)
    user_inputs.append(val)

# Show normal ranges
with st.expander(f"📊 Normal Ranges for {selected_disease.capitalize()} Features"):
    if selected_disease in NORMAL_RANGES:
        for feat in features:
            range_text = NORMAL_RANGES[selected_disease].get(feat, "🔎 Range not specified")
            st.markdown(f"- **{feat}**: {range_text}")
    else:
        st.info("Ranges not available for this disease yet.")

# Prediction button
if st.button("Predict"):
    result = predict_disease(selected_disease, user_inputs)
    if result == 1:
        st.error(f"⚠️ Prediction: Positive for {selected_disease.capitalize()} Disease")
    else:
        st.success(f"✅ Prediction: Negative for {selected_disease.capitalize()} Disease")
