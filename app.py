import os
import sys
import subprocess
from pathlib import Path
import streamlit as st
import pandas as pd
import plotly.express as px
import joblib

from analysis import load_data, get_summary

# ---------------- PAGE CONFIG ----------------

st.set_page_config(
    page_title="Student Success Predictor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------- CUSTOM CSS ----------------

st.markdown("""
<style>
.stApp {
    background: linear-gradient(
        135deg, #0b1020 0%, #17152f 55%, #10172a 100%
    );
    color: #f4f5ff;
}

.block-container {
    max-width: 1450px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}

[data-testid="stSidebar"] {
    background: #111326;
    border-right: 1px solid #2b2c4c;
}

h1, h2, h3 {
    color: #f5f3ff !important;
}

p, label {
    color: #d5d8ed;
}

[data-testid="stMetric"] {
    background: rgba(255,255,255,0.055);
    border: 1px solid rgba(167,139,250,0.24);
    border-radius: 16px;
    padding: 20px;
}

[data-testid="stMetricValue"] {
    color: #c4b5fd;
}

.stButton > button,
.stFormSubmitButton > button {
    background: linear-gradient(90deg, #6366f1, #a855f7);
    color: white;
    border: none;
    border-radius: 11px;
    min-height: 44px;
    font-weight: 600;
}

div[data-testid="stPlotlyChart"] {
    background: rgba(255,255,255,0.025);
    border-radius: 16px;
}

[data-testid="stAlert"] {
    border-radius: 12px;
}
</style>
""", unsafe_allow_html=True)


# ---------------- LOAD FILES ----------------


# Project directory: works locally and on Streamlit Cloud
BASE_DIR = Path(__file__).resolve().parent

DATA_PATH = BASE_DIR / "student_data.csv"
MODEL_PATH = BASE_DIR / "student_model.pkl"


def run_script(script_name):
    """Run a project script on the current machine."""
    script_path = BASE_DIR / script_name

    subprocess.run(
        [sys.executable, str(script_path)],
        cwd=str(BASE_DIR),
        check=True
    )


@st.cache_data
def get_data():
    # Generate the dataset if it does not exist
    if not DATA_PATH.exists():
        run_script("generate_data.py")

    return load_data(str(DATA_PATH))


@st.cache_resource
def get_model():
    # Generate training data if necessary
    if not DATA_PATH.exists():
        run_script("generate_data.py")

    # Train the model if it does not exist
    if not MODEL_PATH.exists():
        run_script("train_model.py")

    return joblib.load(str(MODEL_PATH))



try:
    df = get_data()
    summary = get_summary(df)
except Exception as error:
    st.error(f"Could not load student data: {error}")
    st.info("Run: python generate_data.py")
    st.stop()


# ---------------- SIDEBAR ----------------

with st.sidebar:
    st.markdown("# 🎓 Student Success")
    st.caption("ACADEMIC INTELLIGENCE")

    page = st.radio(
        "NAVIGATION",
        [
            "Dashboard",
            "Predict Result",
            "Analytics",
            "Insights"
        ]
    )

    st.divider()
    st.caption("Student Performance Analytics")
    st.caption("Python + Machine Learning")


# ---------------- HEADER ----------------

st.title("Student Success Predictor")
st.caption(
    "Understand performance. Explore patterns. "
    "Support better academic outcomes."
)


# ---------------- DASHBOARD ----------------

if page == "Dashboard":

    st.subheader("Academic Overview")
    st.caption("Summary of historical student records")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Students Analyzed",
        f"{summary['total_students']:,}"
    )

    c2.metric(
        "Historical Pass Rate",
        f"{summary['pass_rate']:.1f}%"
    )

    c3.metric(
        "Historical Fail Count",
        f"{summary['fail_count']:,}"
    )

    c4.metric(
        "Average Attendance",
        f"{summary['avg_attendance']:.1f}%"
    )

    st.write("")

    left, right = st.columns(2)

    with left:
        result_counts = (
            df["result"]
            .value_counts()
            .reindex(["Pass", "Fail"], fill_value=0)
            .rename_axis("Result")
            .reset_index(name="Students")
        )

        fig = px.pie(
            result_counts,
            names="Result",
            values="Students",
            hole=0.65,
            title="Historical Pass / Fail Distribution",
            color="Result",
            color_discrete_map={
                "Pass": "#8b5cf6",
                "Fail": "#fb7185"
            },
            template="plotly_dark"
        )

        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)"
        )

        st.plotly_chart(fig, use_container_width=True)

    with right:
        fig = px.histogram(
            df,
            x="attendance",
            color="result",
            barmode="overlay",
            opacity=0.8,
            title="Attendance Distribution",
            color_discrete_map={
                "Pass": "#8b5cf6",
                "Fail": "#fb7185"
            },
            template="plotly_dark"
        )

        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)"
        )

        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Recent Student Records")
    st.dataframe(
        df.tail(10),
        use_container_width=True,
        hide_index=True
    )

    st.caption(
        "These charts describe the dataset's historical labels, "
        "not future prediction results."
    )


# ---------------- PREDICT RESULT ----------------

elif page == "Predict Result":

    st.subheader("Individual Student Assessment")

    st.write(
        "Enter the student's academic information "
        "to estimate their likely result."
    )

    with st.form("student_prediction_form"):

        student_id = st.text_input(
            "Student ID",
            placeholder="e.g. STU501"
        )

        col1, col2 = st.columns(2)

        with col1:
            attendance = st.slider(
                "Attendance (%)",
                min_value=0,
                max_value=100,
                value=80
            )

            internal_marks = st.number_input(
                "Internal Marks (out of 30)",
                min_value=0,
                max_value=30,
                value=22
            )

            study_hours = st.number_input(
                "Daily Study Hours",
                min_value=0.0,
                max_value=16.0,
                value=3.0,
                step=0.5
            )

        with col2:
            previous_score = st.number_input(
                "Previous Exam Score (%)",
                min_value=0,
                max_value=100,
                value=65
            )

            assignments = st.slider(
                "Assignments Completed (out of 10)",
                min_value=0,
                max_value=10,
                value=8
            )

        submitted = st.form_submit_button(
            "✨ Predict Result",
            use_container_width=True
        )

    if submitted:

        try:
            model = get_model()

            student = pd.DataFrame([{
                "attendance": attendance,
                "internal_marks": internal_marks,
                "study_hours": study_hours,
                "previous_score": previous_score,
                "assignments_completed": assignments
            }])

            prediction = model.predict(student)[0]

            classes = list(model.classes_)
            probabilities = model.predict_proba(student)[0]

            pass_probability = (
                float(probabilities[classes.index("Pass")]) * 100
            )

            st.divider()
            st.subheader(
                f"Prediction for {student_id.strip() or 'Student'}"
            )

            a, b = st.columns(2)

            with a:
                if prediction == "Pass":
                    st.success("Predicted Outcome: PASS")
                else:
                    st.error("Predicted Outcome: FAIL")

            with b:
                st.metric(
                    "Estimated Pass Probability",
                    f"{pass_probability:.1f}%"
                )

            st.progress(
                min(1.0, max(0.0, pass_probability / 100))
            )

            st.caption(
                "This probability is a model estimate, "
                "not a guarantee of the student's actual result."
            )

            suggestions = []

            if attendance < 75:
                suggestions.append(
                    "Review attendance and create a realistic "
                    "plan for attending upcoming classes."
                )

            if internal_marks < 18:
                suggestions.append(
                    "Review internal assessment topics and "
                    "practice questions with faculty guidance."
                )

            if study_hours < 2:
                suggestions.append(
                    "Try a consistent, manageable daily study routine."
                )

            if assignments < 7:
                suggestions.append(
                    "Prioritize incomplete assignments and "
                    "ask for help with difficult topics."
                )

            if suggestions:
                st.subheader("Suggested Next Steps")

                for suggestion in suggestions:
                    st.markdown(f"- {suggestion}")
            else:
                st.info(
                    "Continue reviewing progress and maintaining "
                    "consistent study habits."
                )

        except FileNotFoundError:
            st.error(
                "Model not found. Run: python train_model.py"
            )

        except Exception as error:
            st.error(f"Prediction failed: {error}")
            
            
    st.divider()
    st.subheader("Bulk Student Prediction")
    st.write("Upload a CSV with the required student features.")

    uploaded_file = st.file_uploader(
        "Upload student CSV",
        type=["csv"],
        key="bulk_student_csv"
    )

    if uploaded_file is not None:
        try:
            bulk_df = pd.read_csv(uploaded_file)

            required_features = [
                "attendance",
                "internal_marks",
                "study_hours",
                "previous_score",
                "assignments_completed"
            ]

            missing = set(required_features) - set(bulk_df.columns)

            if missing:
                st.error(
                    f"Missing columns: {sorted(missing)}"
                )
            else:
                # Convert features to numeric values
                bulk_X = bulk_df[required_features].apply(
                    pd.to_numeric, errors="raise"
                )

                # Check input ranges
                valid = (
                    bulk_X["attendance"].between(0, 100)
                    & bulk_X["internal_marks"].between(0, 30)
                    & bulk_X["study_hours"].between(0, 16)
                    & bulk_X["previous_score"].between(0, 100)
                    & bulk_X["assignments_completed"].between(0, 10)
                )

                if not valid.all():
                    st.error(
                        "Some rows contain values outside the "
                        "allowed ranges. Please correct the CSV."
                    )
                elif bulk_X.isnull().any().any():
                    st.error("The CSV contains missing values.")
                else:
                    bulk_model = get_model()

                    results = bulk_model.predict(bulk_X)
                    probabilities = bulk_model.predict_proba(bulk_X)
                    classes = list(bulk_model.classes_)

                    bulk_df["predicted_result"] = results
                    bulk_df["pass_probability_percent"] = (
                        probabilities[:, classes.index("Pass")] * 100
                    ).round(1)

                    st.success(
                        f"Processed {len(bulk_df)} student records."
                    )

                    st.dataframe(
                        bulk_df,
                        use_container_width=True,
                        hide_index=True
                    )

                    st.download_button(
                        "⬇ Download Predictions",
                        data=bulk_df.to_csv(index=False).encode("utf-8"),
                        file_name="student_predictions.csv",
                        mime="text/csv"
                    )

        except (ValueError, KeyError) as error:
            st.error(f"Please check the CSV data: {error}")



# ---------------- ANALYTICS ----------------

elif page == "Analytics":

    st.subheader("Explore Academic Performance")

    selected_result = st.selectbox(
        "Filter by result",
        ["All", "Pass", "Fail"]
    )

    filtered = df.copy()

    if selected_result != "All":
        filtered = filtered[
            filtered["result"] == selected_result
        ]

    fig = px.scatter(
        filtered,
        x="attendance",
        y="previous_score",
        color="result",
        size="internal_marks",
        hover_data=[
            "study_hours",
            "assignments_completed"
        ],
        title="Attendance vs Previous Score",
        color_discrete_map={
            "Pass": "#8b5cf6",
            "Fail": "#fb7185"
        },
        template="plotly_dark"
    )

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)"
    )

    st.plotly_chart(fig, use_container_width=True)

    fig2 = px.box(
        filtered,
        x="result",
        y="internal_marks",
        color="result",
        title="Internal Marks by Historical Result",
        template="plotly_dark"
    )

    fig2.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)"
    )

    st.plotly_chart(fig2, use_container_width=True)

    st.download_button(
        "⬇ Download Filtered Data",
        data=filtered.to_csv(index=False).encode("utf-8"),
        file_name="student_analytics.csv",
        mime="text/csv"
    )


# ---------------- INSIGHTS ----------------

elif page == "Insights":

    st.subheader("Academic Improvement Insights")

    low_attendance = int(
        (df["attendance"] < 75).sum()
    )

    low_internal = int(
        (df["internal_marks"] < 18).sum()
    )

    low_assignments = int(
        (df["assignments_completed"] < 7).sum()
    )

    c1, c2, c3 = st.columns(3)

    c1.metric("Attendance Below 75%", low_attendance)
    c2.metric("Internal Marks Below 18", low_internal)
    c3.metric("Fewer Than 7 Assignments", low_assignments)

    st.info(
        "These are simple screening indicators for further review, "
        "not proof that a student will fail."
    )

    st.markdown("### Suggested Support Strategies")

    st.markdown("""
    - **Attendance:** identify scheduling barriers and support
      consistent class participation.
    - **Internal assessments:** arrange revision sessions and
      practice exercises.
    - **Assignments:** break pending work into smaller tasks.
    - **Study routine:** encourage a realistic and consistent
      study schedule.
    - **Follow-up:** review progress with the student and faculty.
    """)

    st.caption(
        "Insights are based on simple thresholds, not causal findings."
    )
