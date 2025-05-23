import streamlit as st
import multi_disease_model as mdm

st.set_page_config(page_title="Multi Disease Prediction", layout="centered")
st.title("🩺 Multiple Disease Prediction System")

# Sidebar
st.sidebar.title("Select Disease")
choice = st.sidebar.radio("", [
    "Diabetes Prediction",
    "Heart Disease Prediction",
    "Kidney Disease Prediction",
    "Liver Disease Prediction"
])

# Load models only once
if 'loaded' not in st.session_state:
    try:
        mdm.load_and_train_all()
        st.session_state.loaded = True
    except FileNotFoundError as e:
        st.sidebar.error(str(e))
        st.stop()

# Normal ranges dictionary
NORMAL_RANGES = {
    "diabetes": {
        "Pregnancies": "0–6",
        "Glucose": "70–140 mg/dL",
        "BloodPressure": "60–90 mmHg",
        "SkinThickness": "10–50 mm",
        "Insulin": "15–276 uU/mL",
        "BMI": "18.5–24.9",
        "DiabetesPedigreeFunction": "0.0–2.5",
        "Age": "20–60"
    },
    "heart": {
        "age": "30–70",
        "sex": "0 = female, 1 = male",
        "cp": "0–3",
        "trestbps": "90–140 mmHg",
        "chol": "125–200 mg/dL",
        "fbs": "0 = false, 1 = true",
        "restecg": "0–2",
        "thalach": "100–200 bpm",
        "exang": "0 = no, 1 = yes",
        "oldpeak": "0–2.0",
        "slope": "0–2",
        "ca": "0–4",
        "thal": "0, 1, 2"
    },
    "kidney": {
        "age": "15–90",
        "bp": "60–90",
        "sg": "1.005–1.025",
        "al": "0–3",
        "su": "0–5",
        "rbc": "0 = normal, 1 = abnormal",
        "pc": "0 = normal, 1 = abnormal",
        "pcc": "0 = not present, 1 = present",
        "ba": "0 = not present, 1 = present",
        "bgr": "70–140",
        "bu": "7–20",
        "sc": "0.5–1.5",
        "sod": "135–145",
        "pot": "3.5–5.5",
        "hemo": "12–17",
        "pcv": "36–50",
        "wc": "4000–11000",
        "rc": "4.5–6.0",
        "htn": "0 = no, 1 = yes",
        "dm": "0 = no, 1 = yes",
        "cad": "0 = no, 1 = yes",
        "appet": "0 = good, 1 = poor",
        "pe": "0 = no, 1 = yes",
        "ane": "0 = no, 1 = yes"
    },
    "liver": {
        "Age": "20–80",
        "Gender": "0 = Female, 1 = Male",
        "Total_Bilirubin": "0.3–1.2",
        "Direct_Bilirubin": "0–0.3",
        "Alkaline_Phosphotase": "44–147",
        "Alamine_Aminotransferase": "7–56",
        "Aspartate_Aminotransferase": "10–40",
        "Total_Protiens": "6.0–8.3",
        "Albumin": "3.4–5.4",
        "Albumin_and_Globulin_Ratio": "0.8–2.0"
    }
}

# Render form with ranges
def render_form(key_name, disease_key):
    st.header(f"{disease_key.capitalize()} Prediction")
    features = mdm.get_feature_info(disease_key)
    inputs = []

    with st.form(key_name):
        for feat in features:
            label = feat
            if feat in NORMAL_RANGES.get(disease_key, {}):
                label = f"{feat} (Normal: {NORMAL_RANGES[disease_key][feat]})"
            val = st.number_input(label=label, value=0.0, key=feat)
            inputs.append(val)
        submit = st.form_submit_button("Predict")
    
    if submit:
        prediction = mdm.predict_disease(disease_key, inputs)
        if prediction == 1:
            st.error(f"⚠️ Prediction: Positive for {disease_key.capitalize()} Disease")
        else:
            st.success(f"✅ Prediction: Negative for {disease_key.capitalize()} Disease")

# Routing based on sidebar selection
if choice == "Diabetes Prediction":
    render_form("form_diabetes", "diabetes")
elif choice == "Heart Disease Prediction":
    render_form("form_heart", "heart")
elif choice == "Kidney Disease Prediction":
    render_form("form_kidney", "kidney")
elif choice == "Liver Disease Prediction":
    render_form("form_liver", "liver")
