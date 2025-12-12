import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(page_title="AI Student Performance Predictor", layout="wide")
st.title("🎓 AI Student Performance Predictor")
st.markdown("Predict student grade + get AI-generated summary using Gemini.")

# -----------------------------------------------------
# 1. LOAD + TRAIN MODEL
# -----------------------------------------------------
@st.cache_resource
def load_and_train(path="masked_data.csv"):
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()

    # Normalize categorical
    df["Gender"] = df["Gender"].map({"Male": 0, "Female": 1})

    yes_no = {"Yes": 1, "No": 0}
    df["Extracurricular_Activities"] = df["Extracurricular_Activities"].map(yes_no)
    df["Internet_Access_at_Home"] = df["Internet_Access_at_Home"].map(yes_no)

    edu_map = {"High School": 1, "Bachelor's": 2, "Master's": 3, "PhD": 4}
    df["Parent_Education_Level"] = df["Parent_Education_Level"].map(edu_map)

    income_map = {"Low": 1, "Medium": 2, "High": 3}
    df["Family_Income_Level"] = df["Family_Income_Level"].map(income_map)

    # ------------------ FIXED ONE-HOT ENCODING ------------------
    if "Department" in df.columns:
        df = pd.get_dummies(df, columns=["Department"], prefix="Department")
    else:
        df["Department_CS"] = 0
        df["Department_Engineering"] = 0
        df["Department_Mathematics"] = 0

    # Fill NaN
    df = df.apply(pd.to_numeric, errors="coerce")
    df.fillna(df.mean(), inplace=True)

    # Encode target
    label_encoder = LabelEncoder()
    df["Grade"] = label_encoder.fit_transform(df["Grade"])

    # Prepare X, y
    X = df.drop(columns=["Grade"])
    y = df["Grade"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X_scaled, y)

    return model, scaler, label_encoder, X.columns.tolist(), df


model, scaler, label_encoder, model_columns, df_ref = load_and_train()
st.success("Model trained successfully.")


# -----------------------------------------------------
# 2. SIDEBAR — USER INPUT
# -----------------------------------------------------
st.sidebar.header("Fill Student Details")

api_key = st.sidebar.text_input("Gemini API Key", type="password")

gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
dept = st.sidebar.selectbox("Department", ["CS", "Engineering", "Mathematics"])
attendance = st.sidebar.slider("Attendance (%)", 0, 100, 75)
study_hours = st.sidebar.number_input("Study Hours / Week", 0, 50, 10)
stress = st.sidebar.slider("Stress Level (1-10)", 1, 10, 5)
sleep = st.sidebar.number_input("Sleep Hours / Night", 0, 12, 7)

midterm = st.sidebar.number_input("Midterm Score", 0, 100, 70)
final = st.sidebar.number_input("Final Score", 0, 100, 80)
assignments = st.sidebar.number_input("Assignments Avg", 0, 100, 75)
projects = st.sidebar.number_input("Projects Score", 0, 100, 85)

extracurricular = st.sidebar.selectbox("Extracurricular Activities", ["Yes", "No"])
internet = st.sidebar.selectbox("Internet Access at Home", ["Yes", "No"])
parent_edu = st.sidebar.selectbox("Parent Education Level", ["High School", "Bachelor's", "Master's", "PhD"])
income = st.sidebar.selectbox("Family Income Level", ["Low", "Medium", "High"])

quizzes = df_ref["Quizzes_Avg"].mean()
participation = df_ref["Participation_Score"].mean()
age = df_ref["Age"].mean()
total_score = midterm + final + assignments + projects + quizzes + participation

raw_input = {
    "Gender": gender,
    "Age": age,
    "Attendance (%)": attendance,
    "Midterm_Score": midterm,
    "Final_Score": final,
    "Assignments_Avg": assignments,
    "Quizzes_Avg": quizzes,
    "Participation_Score": participation,
    "Projects_Score": projects,
    "Total_Score": total_score,
    "Study_Hours_per_Week": study_hours,
    "Extracurricular_Activities": extracurricular,
    "Internet_Access_at_Home": internet,
    "Parent_Education_Level": parent_edu,
    "Family_Income_Level": income,
    "Stress_Level (1-10)": stress,
    "Sleep_Hours_per_Night": sleep,
    "Department": dept
}

st.subheader("Your Input")
st.dataframe(pd.DataFrame([raw_input]))


# -----------------------------------------------------
# 3. PREPROCESS INPUT
# -----------------------------------------------------
def preprocess_for_model(d):
    df = pd.DataFrame([d])

    df["Gender"] = df["Gender"].map({"Male": 0, "Female": 1})
    df["Extracurricular_Activities"] = df["Extracurricular_Activities"].map({"Yes": 1, "No": 0})
    df["Internet_Access_at_Home"] = df["Internet_Access_at_Home"].map({"Yes": 1, "No": 0})

    edu_map = {"High School": 1, "Bachelor's": 2, "Master's": 3, "PhD": 4}
    df["Parent_Education_Level"] = df["Parent_Education_Level"].map(edu_map)

    income_map = {"Low": 1, "Medium": 2, "High": 3}
    df["Family_Income_Level"] = df["Family_Income_Level"].map(income_map)

    df["Department_CS"] = 1 if d["Department"] == "CS" else 0
    df["Department_Engineering"] = 1 if d["Department"] == "Engineering" else 0
    df["Department_Mathematics"] = 1 if d["Department"] == "Mathematics" else 0

    df = df.drop(columns=["Department"])

    # Add missing columns with 0
    for col in model_columns:
        if col not in df.columns:
            df[col] = 0

    df = df[model_columns]
    df = df.apply(pd.to_numeric, errors="coerce").fillna(0)

    df_scaled = scaler.transform(df)
    return df_scaled


# -----------------------------------------------------
# 4. PREDICT + AI SUMMARY
# -----------------------------------------------------
if st.button("🔮 Predict Grade & Generate Summary"):
    X = preprocess_for_model(raw_input)

    encoded_pred = model.predict(X)[0]
    pred_label = label_encoder.inverse_transform([encoded_pred])[0]

    st.success(f"🎯 **Predicted Grade: {pred_label}**")

    if not api_key:
        st.warning("Enter your Gemini API key to get the AI summary.")
        st.stop()

    # ---------- Gemini API ----------
    try:
        import google.generativeai as genai
        genai.configure(api_key=api_key)

        model_g = genai.GenerativeModel("gemini-2.0-flash-lite-preview-02-05")

        prompt = f"""
Student Details:
{raw_input}

Predicted Grade: {pred_label}

Generate a short summary with:
- Strengths  
- Weaknesses  
- 4 actionable suggestions  
"""

        response = model_g.generate_content(prompt)

        st.subheader("🧠 AI Summary")
        st.write(response.text)

    except Exception as e:
        st.error(f"Gemini Error: {e}")
