import streamlit as st
import pandas as pd
from gemini_agent import GeminiAgent

# ------------------------------
# Load Dataset
# ------------------------------
CSV_PATH = "masked_data.csv"  # <- your actual file name

st.title("📊 Student Performance Analysis (with Gemini AI Summary)")

@st.cache_data
def load_dataset():
    return pd.read_csv(CSV_PATH)

df = load_dataset()
st.success("Dataset loaded successfully!")

# Expected columns
expected_cols = [
    'Gender', 'Age', 'Attendance (%)', 'Midterm_Score', 'Final_Score',
    'Assignments_Avg', 'Quizzes_Avg', 'Participation_Score',
    'Projects_Score', 'Total_Score', 'Grade', 'Study_Hours_per_Week',
    'Extracurricular_Activities', 'Internet_Access_at_Home',
    'Parent_Education_Level', 'Family_Income_Level',
    'Stress_Level (1-10)', 'Sleep_Hours_per_Night',
    'Department_CS', 'Department_Engineering', 'Department_Mathematics'
]

missing = [c for c in expected_cols if c not in df.columns]
if missing:
    st.warning(
        f"⚠ Missing columns in dataset: {missing}\n"
        "App will continue using available columns."
    )

# ------------------------------
# Input Section
# ------------------------------
st.header("✍️ Enter Student Data for Analysis")

gender = st.selectbox("Gender", ["Male", "Female"])
age = st.number_input("Age", 10, 30, 20)
attendance = st.number_input("Attendance (%)", 0, 100, 80)
midterm = st.number_input("Midterm Score", 0, 100, 75)
final = st.number_input("Final Score", 0, 100, 80)
assign_avg = st.number_input("Assignment Average", 0, 100, 70)
quiz_avg = st.number_input("Quiz Average", 0, 100, 65)
participation = st.number_input("Participation Score", 0, 100, 60)
project = st.number_input("Project Score", 0, 100, 85)
study_hours = st.number_input("Study Hours/Week", 0, 80, 10)
extra = st.selectbox("Extracurricular Activities", ["Yes", "No"])
internet = st.selectbox("Internet Access at Home", ["Yes", "No"])
parent_edu = st.selectbox("Parent Education Level", ["Low", "Medium", "High"])
income = st.selectbox("Family Income Level", ["Low", "Medium", "High"])
stress = st.slider("Stress Level (1-10)", 1, 10, 5)
sleep = st.slider("Sleep Hours/Night", 0, 12, 7)

department = st.selectbox("Department", ["CS", "Engineering", "Mathematics"])

# One-hot encoding for department
dept_cs = 1 if department == "CS" else 0
dept_eng = 1 if department == "Engineering" else 0
dept_math = 1 if department == "Mathematics" else 0

# Build dataframe
input_df = pd.DataFrame([{
    "Gender": 1 if gender == "Male" else 0,
    "Age": age,
    "Attendance (%)": attendance,
    "Midterm_Score": midterm,
    "Final_Score": final,
    "Assignments_Avg": assign_avg,
    "Quizzes_Avg": quiz_avg,
    "Participation_Score": participation,
    "Projects_Score": project,
    "Total_Score": (midterm + final + project + assign_avg) / 4,
    "Study_Hours_per_Week": study_hours,
    "Extracurricular_Activities": 1 if extra == "Yes" else 0,
    "Internet_Access_at_Home": 1 if internet == "Yes" else 0,
    "Parent_Education_Level": ["Low", "Medium", "High"].index(parent_edu),
    "Family_Income_Level": ["Low", "Medium", "High"].index(income),
    "Stress_Level (1-10)": stress,
    "Sleep_Hours_per_Night": sleep,
    "Department_CS": dept_cs,
    "Department_Engineering": dept_eng,
    "Department_Mathematics": dept_math
}])

st.subheader("🔍 Your Entered Data")
st.dataframe(input_df)

# ------------------------------
# Gemini AI Summary Section
# ------------------------------
st.header("🤖 AI Summary (Powered by Gemini)")

api_key = st.text_input("Enter your Gemini API key", type="password")
st.caption("🔐 Your API key is never stored.")

summ_btn = st.button("Generate AI Summary")

if summ_btn:
    if not api_key:
        st.error("❌ Please enter your API key.")
    else:
        agent = GeminiAgent(api_key)
        raw_text = input_df.to_dict(orient="records")[0]

        with st.spinner("Thinking..."):
            summary = agent.summarize(str(raw_text))

        st.subheader("📘 AI Summary")
        st.write(summary)
