import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from gemini_agent import GeminiAgent

st.set_page_config(page_title="Student Performance Prediction", layout="wide")

# --------------------------------------------------------------------
# LOAD DATA + PREPROCESS
# --------------------------------------------------------------------
@st.cache_data
def load_preprocessed_data():
    df = pd.read_csv("masked_data.csv")

    # Identify categorical columns
    categorical_cols = df.select_dtypes(include="object").columns

    encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        encoders[col] = le

    # Save means for auto-filling missing features
    feature_means = df.mean(numeric_only=True).to_dict()

    return df, encoders, feature_means


df, encoders, feature_means = load_preprocessed_data()

# --------------------------------------------------------------------
# TRAIN MODEL
# --------------------------------------------------------------------
@st.cache_resource
def train_model(df):
    X = df.drop("Grade", axis=1)
    y = df["Grade"]

    model = RandomForestClassifier(n_estimators=250, random_state=42)
    model.fit(X, y)

    return model, X.columns


model, feature_cols = train_model(df)

st.success("✔ Model trained successfully!")


# --------------------------------------------------------------------
# USER-FRIENDLY UI INPUT FORM
# --------------------------------------------------------------------
st.header("🎓 Enter Student Details")

col1, col2 = st.columns(2)

with col1:
    gender = st.selectbox("Gender", ["Male", "Female", "Other"])
    department = st.selectbox("Department", ["CS", "Engineering", "Mathematics"])
    attendance = st.number_input("Attendance (%)", 0, 100, 75)
    midterm = st.number_input("Midterm Score", 0, 100, 50)
    final_score = st.number_input("Final Score", 0, 100, 50)

with col2:
    assignments = st.number_input("Assignments Average", 0, 100, 70)
    projects = st.number_input("Projects Score", 0, 100, 70)
    study_hours = st.number_input("Study Hours per Week", 0, 80, 10)
    stress = st.number_input("Stress Level (1-10)", 1, 10, 5)
    sleep = st.number_input("Sleep Hours per Night", 0, 12, 7)

internet = st.selectbox("Internet Access at Home", ["Yes", "No"])
extracurricular = st.selectbox("Extracurricular Activities", ["Yes", "No"])
parent_edu = st.selectbox("Parent Education Level", ["Low", "Medium", "High"])
family_income = st.selectbox("Family Income Level", ["Low", "Medium", "High"])


# --------------------------------------------------------------------
# FEATURE ENGINEERING FOR MODEL INPUT
# --------------------------------------------------------------------
def build_feature_vector():
    inp = {}

    # Encode using original LabelEncoders
    inp["Gender"] = encoders["Gender"].transform([gender])[0]
    inp["Parent_Education_Level"] = encoders["Parent_Education_Level"].transform([parent_edu])[0]
    inp["Family_Income_Level"] = encoders["Family_Income_Level"].transform([family_income])[0]
    inp["Internet_Access_at_Home"] = encoders["Internet_Access_at_Home"].transform([internet])[0]
    inp["Extracurricular_Activities"] = encoders["Extracurricular_Activities"].transform([extracurricular])[0]

    # Numeric features
    inp["Attendance (%)"] = attendance
    inp["Midterm_Score"] = midterm
    inp["Final_Score"] = final_score
    inp["Assignments_Avg"] = assignments
    inp["Projects_Score"] = projects
    inp["Study_Hours_per_Week"] = study_hours
    inp["Stress_Level (1-10)"] = stress
    inp["Sleep_Hours_per_Night"] = sleep

    # Auto-fill missing features from dataset means
    inp["Age"] = feature_means["Age"]
    inp["Quizzes_Avg"] = feature_means["Quizzes_Avg"]
    inp["Participation_Score"] = feature_means["Participation_Score"]

    # Derived feature
    inp["Total_Score"] = midterm + final_score + assignments + projects + feature_means["Quizzes_Avg"]

    # One-hot encoding for Department
    inp["Department_CS"] = 1 if department == "CS" else 0
    inp["Department_Engineering"] = 1 if department == "Engineering" else 0
    inp["Department_Mathematics"] = 1 if department == "Mathematics" else 0

    return pd.DataFrame([inp])


# --------------------------------------------------------------------
# PREDICTION
# --------------------------------------------------------------------
if st.button("Predict Grade"):
    test_df = build_feature_vector()
    pred = model.predict(test_df)[0]

    st.subheader("🎯 Predicted Grade")
    st.metric("Grade", pred)

    # -------------------------------------------------------
    # GEMINI SUMMARY
    # -------------------------------------------------------
    with st.expander("✨ AI Summary (Gemini)"):
        api_key = st.text_input("Enter Gemini API Key", type="password")

        if api_key:
            agent = GeminiAgent(api_key)
            summary = agent.summarize_prediction(test_df.iloc[0].to_dict(), str(pred))
            st.write(summary)
        else:
            st.info("Enter API key to generate AI summary.")


st.caption("Built using RandomForest + Streamlit + Gemini AI")
