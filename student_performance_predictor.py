"""
==============================================================
AI-Based Student Performance Predictor
FYP Project - BSCS
==============================================================
Description:
    This script performs Exploratory Data Analysis (EDA),
    preprocessing, and trains multiple Machine Learning models
    to predict a student's final grade (A-F) based on
    demographic, academic-habit, and lifestyle features.

    Note: Score columns (math/science/english/overall_score)
    are intentionally EXCLUDED from features, since they cause
    data leakage (final_grade is derived directly from them).
    Only "cause" features are used, so the model predicts
    performance BEFORE exam scores exist.

Dataset columns expected:
    student_id, age, gender, school_type, parent_education,
    study_hours, attendance_percentage, internet_access,
    travel_time, extra_activities, study_method,
    math_score, science_score, english_score, overall_score,
    final_grade
==============================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from xgboost import XGBClassifier

try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False


# ==============================================================
# 1. LOAD DATA
# ==============================================================
DATA_PATH = "Student_Performance.csv"   # <-- update path if needed

def load_data(path=DATA_PATH):
    df = pd.read_csv(path)
    print(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    return df


# ==============================================================
# 2. EXPLORATORY DATA ANALYSIS (EDA)
# ==============================================================
def run_eda(df):
    print("\n=== Missing Values ===")
    print(df.isnull().sum())

    print("\n=== Final Grade Distribution ===")
    print(df['final_grade'].value_counts().sort_index())

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    sns.countplot(x='final_grade', data=df, order=sorted(df['final_grade'].unique()), ax=axes[0])
    axes[0].set_title('Final Grade Distribution')

    sns.scatterplot(x='study_hours', y='attendance_percentage', hue='final_grade', data=df, ax=axes[1], alpha=0.5)
    axes[1].set_title('Study Hours vs Attendance (colored by grade)')

    plt.tight_layout()
    plt.savefig('eda_plots.png', dpi=120)
    print("\nSaved EDA plots to eda_plots.png")


# ==============================================================
# 3. PREPROCESSING
# ==============================================================
def preprocess(df):
    df = df.drop('student_id', axis=1)

    cat_cols = ['gender', 'school_type', 'parent_education', 'internet_access',
                'travel_time', 'extra_activities', 'study_method']
    encoders = {}
    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        encoders[col] = le

    target_le = LabelEncoder()
    df['final_grade_encoded'] = target_le.fit_transform(df['final_grade'])

    # IMPORTANT: drop leaky columns (score columns directly determine final_grade)
    leaky_cols = ['overall_score', 'math_score', 'science_score',
                  'english_score', 'final_grade', 'final_grade_encoded']

    X = df.drop(leaky_cols, axis=1)
    y = df['final_grade_encoded']

    return X, y, encoders, target_le


# ==============================================================
# 4. TRAIN & COMPARE MULTIPLE MODELS
# ==============================================================
def compare_models(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    X_scaled_full = scaler.fit_transform(X)

    models = {
        'Logistic Regression': (LogisticRegression(max_iter=1000, random_state=42), True),
        'Decision Tree': (DecisionTreeClassifier(max_depth=10, random_state=42), False),
        'Random Forest': (RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42), False),
        'Gradient Boosting': (GradientBoostingClassifier(n_estimators=100, max_depth=5, random_state=42), False),
        'XGBoost': (XGBClassifier(n_estimators=200, max_depth=6, random_state=42, eval_metric='mlogloss'), False),
        'KNN': (KNeighborsClassifier(n_neighbors=15), True),
    }

    results = []
    trained_models = {}

    for name, (model, needs_scaling) in models.items():
        if needs_scaling:
            model.fit(X_train_scaled, y_train)
            pred = model.predict(X_test_scaled)
            cv = cross_val_score(model, X_scaled_full, y, cv=5)
        else:
            model.fit(X_train, y_train)
            pred = model.predict(X_test)
            cv = cross_val_score(model, X, y, cv=5)

        acc = accuracy_score(y_test, pred)
        results.append({
            'Model': name,
            'Test Accuracy (%)': round(acc * 100, 2),
            'CV Mean Accuracy (%)': round(cv.mean() * 100, 2),
            'CV Std (%)': round(cv.std() * 100, 2),
        })
        trained_models[name] = model
        print(f"{name}: Test Acc={acc*100:.2f}%, CV={cv.mean()*100:.2f}% (+/-{cv.std()*100:.2f}%)")

    results_df = pd.DataFrame(results).sort_values('CV Mean Accuracy (%)', ascending=False)
    results_df.to_csv('model_comparison.csv', index=False)
    print("\n=== Final Comparison Table ===")
    print(results_df.to_string(index=False))

    return trained_models, results_df, (X_test, y_test), scaler


# ==============================================================
# 5. DETAILED EVALUATION OF BEST MODEL
# ==============================================================
def evaluate_best_model(model, X_test, y_test, target_le, model_name="Model"):
    y_pred = model.predict(X_test)

    print(f"\n=== Detailed Report: {model_name} ===")
    print(classification_report(y_test, y_pred, target_names=target_le.classes_))

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=target_le.classes_, yticklabels=target_le.classes_)
    plt.title(f'Confusion Matrix - {model_name}')
    plt.xlabel('Predicted Grade')
    plt.ylabel('Actual Grade')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=120)
    print("Saved confusion_matrix.png")


# ==============================================================
# 6. LLM INTEGRATION (Claude) — Natural-Language Recommendations
# ==============================================================
def get_ai_recommendation(student_info: dict, predicted_grade: str, api_key: str = None) -> str:
    """
    Sends the model's numeric prediction + the student's raw feature
    values to Claude, and asks for a short, personalized, actionable
    recommendation in plain language.

    student_info: dict of the student's raw (un-encoded) feature values,
                  e.g. {'study_hours': 2, 'attendance_percentage': 60,
                        'parent_education': 'high school', ...}
    predicted_grade: the grade letter predicted by the ML model (e.g. 'c')
    api_key: your Anthropic API key. Get one at https://console.anthropic.com
             Can also be set via the ANTHROPIC_API_KEY environment variable.
    """
    if not ANTHROPIC_AVAILABLE:
        return "Error: install the 'anthropic' package first (pip install anthropic)."

    client = anthropic.Anthropic(api_key=api_key)  # uses env var if api_key=None

    prompt = f"""You are an academic advisor. A machine learning model predicts this
student will get grade '{predicted_grade}' (scale: a=best, f=worst) based on
their profile below:

{student_info}

In 3-4 short sentences, written in a warm and encouraging tone:
1. Briefly explain the most likely reason(s) for this predicted grade, based on the data.
2. Give 2 concrete, specific suggestions the student could act on this week to improve.

Keep it concise and practical, no generic advice like 'work hard'."""

    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=300,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.content[0].text


def demo_ai_recommendation(model, X_test, target_le, encoders, original_df, api_key=None):
    """
    Picks one sample student from the test set, predicts their grade,
    and prints an AI-generated recommendation for them.
    """
    sample_idx = X_test.index[0]
    sample_features = X_test.loc[[sample_idx]]
    predicted_encoded = model.predict(sample_features)[0]
    predicted_grade = target_le.inverse_transform([predicted_encoded])[0]

    # Decode the sample back to human-readable values for the prompt
    readable = {}
    for col in sample_features.columns:
        val = sample_features.iloc[0][col]
        if col in encoders:
            val = encoders[col].inverse_transform([int(val)])[0]
        readable[col] = val

    print(f"\n=== Sample Student Profile ===\n{readable}")
    print(f"\nPredicted Grade: {predicted_grade}")

    print("\n=== AI-Generated Recommendation ===")
    try:
        advice = get_ai_recommendation(readable, predicted_grade, api_key=api_key)
        print(advice)
    except Exception as e:
        print(f"Could not reach Claude API: {e}")
        print("Make sure ANTHROPIC_API_KEY is set, or pass api_key= explicitly.")


# ==============================================================
# 7. MAIN PIPELINE
# ==============================================================
def main():
    df = load_data()
    run_eda(df)

    X, y, encoders, target_le = preprocess(df)
    trained_models, results_df, (X_test, y_test), scaler = compare_models(X, y)

    best_model_name = results_df.iloc[0]['Model']
    best_model = trained_models[best_model_name]
    print(f"\nBest model: {best_model_name}")

    # Use scaled test set if best model needed scaling
    if best_model_name in ['Logistic Regression', 'KNN']:
        X_test_eval = scaler.transform(X_test)
    else:
        X_test_eval = X_test

    evaluate_best_model(best_model, X_test_eval, y_test, target_le, best_model_name)

    # Save best model + encoders for later use (e.g. in a web app)
    joblib.dump(best_model, 'best_model.pkl')
    joblib.dump(encoders, 'encoders.pkl')
    joblib.dump(target_le, 'target_encoder.pkl')
    print("\nSaved best_model.pkl, encoders.pkl, target_encoder.pkl")

    # --- LLM recommendation demo ---
    # Requires: pip install anthropic, and an API key from https://console.anthropic.com
    # Set it as an environment variable:  export ANTHROPIC_API_KEY="your-key-here"
    demo_ai_recommendation(best_model, X_test, target_le, encoders, df)


if __name__ == "__main__":
    main()
