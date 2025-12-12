# app.py
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import NotFittedError

# Try to import user-provided agent wrapper; if missing we'll call Gemini directly
try:
    from agent import GeminiAgent  # user said they will have agent.py
    HAVE_AGENT_WRAPPER = True
except Exception:
    HAVE_AGENT_WRAPPER = False

# We'll lazily import google.generativeai only when needed to avoid import errors at startup.
# requirements.txt must include: google-generativeai (or adjust for your environment)

st.set_page_config(page_title="AI Student Performance Predictor", layout="wide")
st.title("🎓 AI Student Performance Predictor")
st.markdown("Predict student grade and get an AI-generated summary (Gemini).")

# -----------------------------
# Constants: expected exact columns
# -----------------------------
EXPECTED_COLUMNS = [
 'Gender','Age','Attendance (%)','Midterm_Score','Final_Score','Assignments_Avg',
 'Quizzes_Avg','Participation_Score','Projects_Score','Total_Score','Grade',
 'Study_Hours_per_Week','Extracurricular_Activities','Internet_Access_at_Home',
 'Parent_Education_Level','Family_Income_Level','Stress_Level (1-10)',
 'Sleep_Hours_per_Night','Department_CS','Department_Engineering','Department_Mathematics'
]

# -----------------------------
# Load dataset and train model
# -----------------------------
@st.cache_resource
def load_and_train_model(path="masked_data.csv"):
    # 1. Load
    df = pd.read_csv(path)
    # normalize column names (strip)
    df.columns = df.columns.str.strip()

    # 2. Ensure expected columns present (we won't fail; we'll adapt)
    missing = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    if missing:
        st.warning(f"Warning: dataset missing expected columns: {missing}. App will attempt to proceed with available columns.")
    # 3. Basic cleaning: map common categorical variants to canonical forms
    # Gender variants
    if "Gender" in df.columns:
        df["Gender"] = df["Gender"].astype(str).str.strip().replace({
            "M": "Male", "F": "Female", "m": "Male", "f": "Female"
        })

    # Extracurricular & Internet variants
    for col in ["Extracurricular_Activities", "Internet_Access_at_Home"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().replace({
                "yes": "Yes", "no": "No", "Y": "Yes", "N": "No", "y": "Yes", "n": "No",
                "TRUE": "Yes", "FALSE": "No", "true": "Yes", "false": "No"
            })

    # Parent education normalization (some datasets use slightly different names)
    if "Parent_Education_Level" in df.columns:
        df["Parent_Education_Level"] = df["Parent_Education_Level"].astype(str).str.strip().replace({
            "Bachelors": "Bachelor's", "Bachelor": "Bachelor's", "HighSchool": "High School",
            "High-school": "High School"
        })

    # Family income normalization
    if "Family_Income_Level" in df.columns:
        df["Family_Income_Level"] = df["Family_Income_Level"].astype(str).str.strip().replace({
            "low": "Low", "medium": "Medium", "high": "High"
        })

    # 4. Map binary columns to 0/1 where applicable
    binary_map = {"Yes": 1, "No": 0}
    for col in ["Extracurricular_Activities", "Internet_Access_at_Home"]:
        if col in df.columns:
            df[col] = df[col].map(binary_map).astype(float)

    # Gender mapping
    if "Gender" in df.columns:
        df["Gender"] = df["Gender"].map({"Male": 0, "Female": 1}).astype(float)

    # 5. Ordinal maps
    if "Parent_Education_Level" in df.columns:
        edu_map = {"High School": 1, "Bachelor's": 2, "Master's": 3, "PhD": 4}
        df["Parent_Education_Level"] = df["Parent_Education_Level"].map(edu_map).astype(float)

    if "Family_Income_Level" in df.columns:
        income_map = {"Low": 1, "Medium": 2, "High": 3}
        df["Family_Income_Level"] = df["Family_Income_Level"].map(income_map).astype(float)

    # 6. One-hot for department if original 'Department' exists; otherwise assume Department_* present
    if "Department" in df.columns:
        df = pd.get_dummies(df, columns=["Department"], prefix="Department", drop_first=False)
    else:
        # ensure Department_* columns exist so model/training has consistent columns
        for c in ["Department_CS", "Department_Engineering", "Department_Mathematics"]:
            if c not in df.columns:
                df[c] = 0

    # 7. Convert any remaining object columns to numeric where possible
    for c in df.columns:
        if df[c].dtype == "object":
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # 8. Fill numerical NaNs with column mean
    df.fillna(df.mean(numeric_only=True), inplace=True)

    # 9. Ensure Grade exists
    if "Grade" not in df.columns:
        raise ValueError("masked_data.csv must contain a 'Grade' column.")

    # 10. Label encode Grade
    label_enc = LabelEncoder()
    df["Grade"] = label_enc.fit_transform(df["Grade"].astype(str))

    # 11. Prepare features: drop Grade, ensure numeric-only
    X = df.drop(columns=["Grade"])
    # Keep only numeric features for X
    X = X.select_dtypes(include=[np.number])

    # 12. Scale numeric features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X = pd.DataFrame(X_scaled, columns=X.columns)

    # 13. Train RandomForest
    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X, df["Grade"])

    # Return trained components and reference df (original processed df before scaling)
    return model, scaler, label_enc, X.columns.tolist(), df

# Train on startup (cached)
try:
    model, scaler, label_encoder, model_columns, df_ref = load_and_train_model("masked_data.csv")
except Exception as e:
    st.error(f"Failed to load/train model: {e}")
    st.stop()


st.success("Model trained in-memory (ready).")

# -----------------------------
# Helper: build user-facing UI and auto-compute missing fields
# -----------------------------
st.sidebar.header("Student Parameters (fill the form)")

def get_user_input_auto():
    # Human-friendly fields only
    gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
    dept = st.sidebar.selectbox("Department", ["CS", "Engineering", "Mathematics"])
    attendance = st.sidebar.slider("Attendance (%)", 0, 100, 80)
    study_hours = st.sidebar.number_input("Study Hours per Week", 0, 80, 15)
    stress = st.sidebar.slider("Stress Level (1-10)", 1, 10, 5)
    sleep = st.sidebar.number_input("Sleep Hours per Night", 0, 12, 7)

    st.sidebar.subheader("Exam Scores")
    midterm = st.sidebar.number_input("Midterm Score", 0, 100, 70)
    final = st.sidebar.number_input("Final Score", 0, 100, 75)
    assignments = st.sidebar.number_input("Assignments Avg", 0, 100, 80)
    projects = st.sidebar.number_input("Projects Score", 0, 100, 85)

    extracurricular = st.sidebar.selectbox("Extracurricular Activities", ["Yes", "No"])
    internet = st.sidebar.selectbox("Internet Access at Home", ["Yes", "No"])
    parent_edu = st.sidebar.selectbox("Parent Education Level", ["High School", "Bachelor's", "Master's", "PhD"])
    income = st.sidebar.selectbox("Family Income Level", ["Low", "Medium", "High"])

    # Auto-compute fields from df_ref means
    quizzes = df_ref["Quizzes_Avg"].mean() if "Quizzes_Avg" in df_ref.columns else 70.0
    participation = df_ref["Participation_Score"].mean() if "Participation_Score" in df_ref.columns else 75.0
    age = df_ref["Age"].mean() if "Age" in df_ref.columns else 20.0

    total = midterm + final + assignments + projects + quizzes + participation

    user = {
        "Gender": gender,
        "Age": float(age),
        "Attendance (%)": float(attendance),
        "Midterm_Score": float(midterm),
        "Final_Score": float(final),
        "Assignments_Avg": float(assignments),
        "Quizzes_Avg": float(quizzes),
        "Participation_Score": float(participation),
        "Projects_Score": float(projects),
        "Total_Score": float(total),
        "Study_Hours_per_Week": float(study_hours),
        "Extracurricular_Activities": extracurricular,
        "Internet_Access_at_Home": internet,
        "Parent_Education_Level": parent_edu,
        "Family_Income_Level": income,
        "Stress_Level (1-10)": float(stress),
        "Sleep_Hours_per_Night": float(sleep),
        "Department": dept
    }
    return pd.DataFrame([user])

input_df_raw = get_user_input_auto()
st.subheader("Input preview (human-friendly)")
st.dataframe(input_df_raw, use_container_width=True)

# -----------------------------
# Preprocess user input to match training features
# -----------------------------
def preprocess_input(raw_df):
    df = raw_df.copy()

    # map binary and ordinals (same mappings used in training)
    df["Gender"] = df["Gender"].map({"Male": 0, "Female": 1})
    df["Extracurricular_Activities"] = df["Extracurricular_Activities"].map({"No": 0, "Yes": 1})
    df["Internet_Access_at_Home"] = df["Internet_Access_at_Home"].map({"No": 0, "Yes": 1})

    edu_map = {"High School": 1, "Bachelor's": 2, "Master's": 3, "PhD": 4}
    df["Parent_Education_Level"] = df["Parent_Education_Level"].map(edu_map)

    income_map = {"Low": 1, "Medium": 2, "High": 3}
    df["Family_Income_Level"] = df["Family_Income_Level"].map(income_map)

    # One-hot department -> Department_CS, Department_Engineering, Department_Mathematics
    df["Department_CS"] = 0
    df["Department_Engineering"] = 0
    df["Department_Mathematics"] = 0
    chosen = df.loc[0, "Department"]
    if chosen == "CS":
        df.loc[0, "Department_CS"] = 1
    elif chosen == "Engineering":
        df.loc[0, "Department_Engineering"] = 1
    elif chosen == "Mathematics":
        df.loc[0, "Department_Mathematics"] = 1

    # drop the Department column
    df = df.drop(columns=["Department"])

    # Keep only columns that model_columns contains; create missing ones with 0
    for col in model_columns:
        if col not in df.columns:
            df[col] = 0

    df = df[model_columns]  # order columns same as training

    # Ensure numeric dtype
    df = df.apply(pd.to_numeric, errors="coerce")
    df.fillna(0, inplace=True)

    # Scale numeric columns using the scaler fitted on training data
    # scaler expects the same numeric columns used earlier - we trained scaler on all numeric X columns
    try:
        scaled = scaler.transform(df)
        df_scaled = pd.DataFrame(scaled, columns=df.columns)
    except Exception as e:
        # Fallback: if transform fails, try to coerce and proceed without scaling
        st.warning(f"Scaling failed for input: {e}. Proceeding without scaling.")
        df_scaled = df

    return df_scaled

# -----------------------------
# Predict + Gemini summary
# -----------------------------
if st.button("Predict Grade & Generate Summary"):
    processed = preprocess_input(input_df_raw)

    # Predict
    try:
        pred_encoded = model.predict(processed)[0]
    except Exception as e:
        st.error(f"Prediction failed: {e}")
        raise

    pred_label = label_encoder.inverse_transform([pred_encoded])[0]
    st.success(f"🎯 Predicted Grade: **{pred_label}**")

    # Prepare readable input for summary
    readable = input_df_raw.to_dict(orient="records")[0]
    # make sure readable values are simple types
    for k, v in readable.items():
        if isinstance(v, (np.generic, np.ndarray)):
            readable[k] = v.item() if hasattr(v, "item") else v

    # Gemini summary (requires API key)
    api_key = st.sidebar.text_input("Gemini API Key (for AI summary)", type="password")
    if not api_key:
        st.info("Enter Gemini API key in the sidebar to generate the AI summary.")
    else:
        # Use user-supplied agent wrapper if available
        if HAVE_AGENT_WRAPPER:
            try:
                agent = GeminiAgent(api_key)
                summary = agent.get_summary(readable, pred_label)  # wrapper should implement get_summary
                st.subheader("🧠 AI Summary (via agent.py)")
                st.write(summary)
            except Exception as e:
                st.warning(f"Agent wrapper failed: {e}. Falling back to direct Gemini call.")
                HAVE_AGENT_WRAPPER_FALLBACK = True
        else:
            HAVE_AGENT_WRAPPER_FALLBACK = True

        # fallback: direct call to google.generativeai
        if 'HAVE_AGENT_WRAPPER_FALLBACK' in locals() and HAVE_AGENT_WRAPPER_FALLBACK:
            try:
                import google.generativeai as genai
                genai.configure(api_key=api_key)

                prompt = f"""
You are an educational advisor. Here is a student's profile and prediction.
Student profile:
{readable}

Predicted Grade: {pred_label}

Provide a short summary (3-6 bullet points): strengths, weaknesses, and 4 actionable recommendations for the student.
"""
                res = genai.generate_text(model="gemini-2.0-flash-lite-preview-02-05", prompt=prompt)
                summary_text = getattr(res, "text", None) or str(res)
                st.subheader("🧠 AI Summary (Gemini)")
                st.write(summary_text)
            except Exception as e:
                st.error(f"Failed to call Gemini directly: {e}")
                st.write("If you want AI summaries, ensure 'google-generativeai' is installed and the API key is valid.")

