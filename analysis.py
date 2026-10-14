
import pandas as pd


def load_data(path="student_data.csv"):
    df = pd.read_csv(path)

    required = [
        "attendance",
        "internal_marks",
        "study_hours",
        "previous_score",
        "assignments_completed",
        "result"
    ]

    missing = set(required) - set(df.columns)

    if missing:
        raise ValueError(
            f"Dataset is missing columns: {sorted(missing)}"
        )

    df["result"] = (
        df["result"].astype(str).str.strip().str.title()
    )

    if not df["result"].isin(["Pass", "Fail"]).all():
        raise ValueError(
            "Result must contain only Pass or Fail."
        )

    return df


def get_summary(df):
    total_students = len(df)

    pass_count = int(
        df["result"].eq("Pass").sum()
    )

    fail_count = int(
        df["result"].eq("Fail").sum()
    )

    pass_rate = (
        pass_count / total_students * 100
        if total_students else 0
    )

    avg_attendance = (
        df["attendance"].mean()
        if total_students else 0
    )

    return {
        "total_students": total_students,
        "pass_count": pass_count,
        "fail_count": fail_count,
        "pass_rate": pass_rate,
        "avg_attendance": avg_attendance
    }
