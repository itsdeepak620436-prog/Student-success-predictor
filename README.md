# 🎓 Student Success Predictor

### Student Pass/Fail Prediction System Using Python and Machine Learning

A beginner-friendly Data Analysis and Machine Learning project that analyzes student academic performance and predicts whether a student is likely to pass or fail based on attendance, internal marks, study hours, previous exam scores, and assignment completion.

The project features an interactive dashboard built with **Python and Streamlit**, along with data visualizations and a machine learning prediction model.

## ✨ Features

- 📊 **Interactive Dashboard** — View student statistics, historical pass rates, and attendance distribution.
- 🤖 **Pass/Fail Prediction** — Predict student outcomes using Logistic Regression.
- 📈 **Performance Analytics** — Explore attendance, previous scores, and internal marks using interactive charts.
- 💡 **Improvement Insights** — Identify academic indicators that may require additional support.
- 📁 **CSV Upload** — Upload multiple student records for bulk predictions.
- 📥 **Download Results** — Export analytics and prediction results as CSV files.
- 🎨 **Modern UI** — Dark-themed dashboard with purple accents and interactive visualizations.

## 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| Pandas | Data manipulation and analysis |
| NumPy | Numerical operations and sample data generation |
| Scikit-learn | Machine learning and model evaluation |
| Streamlit | Interactive web application |
| Plotly | Interactive charts and visualizations |
| Joblib | Saving and loading the trained model |

## 📂 Project Structure

```text
Student-Success-Predictor/
│
├── app.py                  # Streamlit frontend
├── generate_data.py        # Generates sample student data
├── train_model.py          # Trains and evaluates the model
├── analysis.py             # Data loading and analysis
├── requirements.txt        # Required Python libraries
├── .gitignore              # Excludes unnecessary files
├── README.md               # Project documentation
├── test_students.csv       # Sample CSV for bulk prediction
├── student_data.csv        # Generated dataset
└── student_model.pkl       # Trained model (generated locally)
```

`student_data.csv` and `student_model.pkl` are generated or prepared locally. The trained model is excluded from GitHub by default, so you can regenerate it using the steps below.

## ⚙️ Installation and Setup

### Prerequisites

- Python 3.10 or a compatible Python version
- Git
- Visual Studio Code or another Python editor

### 1. Clone the repository

```bash
git clone https://github.com/itsdeepak620436-prog/Student-success-predictor.git
```

### 2. Open the project directory

```bash
cd Student-success-predictor
```

### 3. Create a virtual environment

```bash
python -m venv venv
```

### 4. Activate the virtual environment

**Windows PowerShell:**

```powershell
.\venv\Scripts\Activate.ps1
```

**Windows Command Prompt:**

```cmd
venv\Scripts\activate.bat
```

### 5. Install dependencies

```bash
pip install -r requirements.txt
```

### 6. Generate the student dataset

```bash
python generate_data.py
```

This creates a sample CSV containing synthetic student records.

### 7. Train the machine learning model

```bash
python train_model.py
```

This trains a Logistic Regression classifier, displays evaluation metrics, and saves the model as `student_model.pkl`.

### 8. Run the application

```bash
streamlit run app.py
```

Open the local URL displayed in the terminal, typically:

```text
http://localhost:8501
```

## 📊 Dataset Features

The model uses the following input features:

| Feature | Description |
|---|---|
| `attendance` | Attendance percentage |
| `internal_marks` | Internal assessment marks out of 30 |
| `study_hours` | Daily study hours |
| `previous_score` | Previous exam score as a percentage |
| `assignments_completed` | Number of completed assignments out of 10 |

The target column is `result`, which contains the historical class label `Pass` or `Fail`.

## 🧠 Machine Learning Workflow

1. Generate or load student data.
2. Inspect and analyze academic performance.
3. Separate input features from the target label.
4. Split records into training and testing sets.
5. Standardize numerical features.
6. Train a Logistic Regression classifier.
7. Evaluate the model using accuracy, precision, recall, F1-score, and a confusion matrix.
8. Save the trained model and use it in the Streamlit application.

## 🧪 Testing Bulk Predictions

The sample file `test_students.csv` should contain these columns:

```csv
attendance,internal_marks,study_hours,previous_score,assignments_completed
85,24,4,72,9
60,14,1.5,45,5
92,27,5,86,10
```

Upload this file through the **Predict Result → Bulk Student Prediction** section to test predictions for multiple students.

## 🚀 Future Improvements

- Compare Logistic Regression with Decision Tree and Random Forest models.
- Evaluate model calibration and performance on real student datasets.
- Add additional academic performance visualizations.
- Improve data validation and error handling.
- Deploy the application online using Streamlit Community Cloud.
- Add secure authentication and privacy-aware student record management.

## ⚠️ Disclaimer

This project is intended for educational and demonstration purposes.

The initial dataset uses synthetic student records and illustrative labels. Performance on synthetic data does not establish real-world predictive accuracy. Predictions are estimates, not guaranteed academic outcomes, and should not be used as the sole basis for decisions about students.

## 👨‍💻 Author

**Deepak Kr Mahto**

GitHub: [@itsdeepak620436-prog](https://github.com/itsdeepak620436-prog)

---

⭐ If you find this project useful, consider starring the repository!
