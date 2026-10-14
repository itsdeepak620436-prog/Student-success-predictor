
import numpy as np
import pandas as pd

# Reproducible sample data
np.random.seed(42)

n = 500

data = pd.DataFrame({
    "student_id": [
        f"STU{i:03d}" for i in range(1, n + 1)
    ],
    "attendance": np.random.randint(35, 101, n),
    "internal_marks": np.random.randint(5, 31, n),
    "study_hours": np.round(
        np.random.uniform(0.5, 8, n), 1
    ),
    "previous_score": np.random.randint(20, 101, n),
    "assignments_completed": np.random.randint(1, 11, n)
})

# Illustrative score used only to create demo labels
score = (
    data["attendance"] * 0.20
    + (data["internal_marks"] / 30) * 100 * 0.35
    + data["previous_score"] * 0.25
    + (data["assignments_completed"] / 10) * 100 * 0.15
    + (data["study_hours"] / 8) * 100 * 0.05
    + np.random.normal(0, 10, n)
)

data["result"] = np.where(score >= 60, "Pass", "Fail")

data.to_csv("student_data.csv", index=False)

print("Dataset created successfully!")
print(data.head())
print("\nResult distribution:")
print(data["result"].value_counts())
print("\nSaved as student_data.csv")
