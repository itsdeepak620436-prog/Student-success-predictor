
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# Load the dataset
df = pd.read_csv("student_data.csv")

FEATURES = [
    "attendance",
    "internal_marks",
    "study_hours",
    "previous_score",
    "assignments_completed"
]

X = df[FEATURES]
y = df["result"]

# Keep a separate test set
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# Standardize values and train the classifier
model = make_pipeline(
    StandardScaler(),
    LogisticRegression(max_iter=1000)
)

model.fit(X_train, y_train)

# Evaluate on unseen test records
predictions = model.predict(X_test)

print("Accuracy:", round(
    accuracy_score(y_test, predictions) * 100, 2
), "%")

print("\nClassification Report:")
print(classification_report(
    y_test,
    predictions,
    zero_division=0
))

print("\nConfusion Matrix:")
print(confusion_matrix(
    y_test,
    predictions,
    labels=["Fail", "Pass"]
))

# Save the complete pipeline
joblib.dump(model, "student_model.pkl")

print("\nModel saved as student_model.pkl")
