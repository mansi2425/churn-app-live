"""
train_model.py
--------------
Connects to the SQLite database, fetches customer data,
performs preprocessing and feature engineering, trains a
Random Forest Classifier, evaluates it, and persists the
model artifact and preprocessing objects.

Usage:
    python src/train_model.py
"""

import sqlite3
import pandas as pd
import numpy as np
import os
import joblib
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection   import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble          import RandomForestClassifier
from sklearn.preprocessing     import LabelEncoder, StandardScaler
from sklearn.metrics           import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, classification_report
)

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH       = os.path.join(BASE_DIR, "churn_data.db")
ARTIFACTS_DIR = os.path.join(BASE_DIR, "artifacts")
os.makedirs(ARTIFACTS_DIR, exist_ok=True)

MODEL_PATH    = os.path.join(ARTIFACTS_DIR, "churn_model.pkl")
ENCODER_PATH  = os.path.join(ARTIFACTS_DIR, "label_encoders.pkl")
SCALER_PATH   = os.path.join(ARTIFACTS_DIR, "scaler.pkl")
FEATURES_PATH = os.path.join(ARTIFACTS_DIR, "feature_columns.pkl")

# ── Step 1: Fetch data from SQLite ────────────────────────────────────────────
def fetch_data(db_path: str) -> pd.DataFrame:
    print(f"[INFO] Connecting to database: {db_path}")
    conn = sqlite3.connect(db_path)
    df   = pd.read_sql_query("SELECT * FROM customers", conn)
    conn.close()
    print(f"[INFO] Fetched {len(df)} rows and {df.shape[1]} columns.")
    return df

# ── Step 2: Preprocessing ─────────────────────────────────────────────────────
def preprocess(df: pd.DataFrame):
    # --- Drop non-predictive ID column
    df = df.drop(columns=["customerID"])

    # --- Fix TotalCharges (whitespace → NaN during CSV parsing)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    # --- Impute missing TotalCharges with median
    median_tc = df["TotalCharges"].median()
    df["TotalCharges"].fillna(median_tc, inplace=True)
    print(f"[INFO] Imputed {df['TotalCharges'].isna().sum()} missing TotalCharges with median ({median_tc:.2f})")

    # --- Encode target variable
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

    # --- Identify column types
    categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()
    numerical_cols   = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    numerical_cols   = [c for c in numerical_cols if c != "Churn"]

    print(f"[INFO] Categorical columns ({len(categorical_cols)}): {categorical_cols}")
    print(f"[INFO] Numerical columns  ({len(numerical_cols)}): {numerical_cols}")

    # --- Label-encode all categorical columns
    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le

    # --- Feature Engineering: derive additional meaningful features
    df["AvgMonthlyCharge"]  = df["TotalCharges"] / (df["tenure"] + 1)
    df["ChargePerService"]  = df["MonthlyCharges"] / (
        df[["PhoneService","MultipleLines","OnlineSecurity",
            "OnlineBackup","DeviceProtection","TechSupport",
            "StreamingTV","StreamingMovies"]].sum(axis=1) + 1
    )
    df["TenureGroup"] = pd.cut(
        df["tenure"],
        bins=[0, 12, 24, 48, 72],
        labels=[0, 1, 2, 3],
        include_lowest=True
    ).astype(int)

    return df, label_encoders, numerical_cols, categorical_cols

# ── Step 3: Train / Evaluate ───────────────────────────────────────────────────
def train_and_evaluate(df: pd.DataFrame):
    TARGET   = "Churn"
    features = [c for c in df.columns if c != TARGET]

    X = df[features]
    y = df[TARGET]

    # Scale numerical features
    scaler = StandardScaler()
    X_scaled = X.copy()
    numerical_in_X = [c for c in X.columns if X[c].dtype in ["float64", "int64"]]
    X_scaled[numerical_in_X] = scaler.fit_transform(X[numerical_in_X])

    # Stratified split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f"\n[INFO] Training set: {X_train.shape[0]} | Test set: {X_test.shape[0]}")

    # ── TASK 1: Algorithm Comparison ────────────────────────────
    print("\n" + "="*60)
    print("       ALGORITHM COMPARISON")
    print("="*60)

    from sklearn.linear_model import LogisticRegression
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.svm import SVC
    from sklearn.naive_bayes import GaussianNB

    algorithms = {
        "Logistic Regression" : LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42),
        "Decision Tree"       : DecisionTreeClassifier(class_weight="balanced", random_state=42),
        "K-Nearest Neighbors" : KNeighborsClassifier(),
        "Naive Bayes"         : GaussianNB(),
        "SVM"                 : SVC(class_weight="balanced", probability=True, random_state=42),
        "Random Forest"       : RandomForestClassifier(class_weight="balanced", random_state=42, n_jobs=-1),
    }

    print(f"\n{'Algorithm':<25} {'Accuracy':>10} {'Precision':>11} {'Recall':>9} {'F1':>9} {'ROC-AUC':>10}")
    print("-"*78)

    comparison_results = {}
    for name, model in algorithms.items():
        model.fit(X_train, y_train)
        y_pred      = model.predict(X_test)
        y_prob      = model.predict_proba(X_test)[:, 1]
        acc  = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec  = recall_score(y_test, y_pred)
        f1   = f1_score(y_test, y_pred)
        auc  = roc_auc_score(y_test, y_prob)
        comparison_results[name] = {"Accuracy": acc, "Precision": prec, "Recall": rec, "F1": f1, "AUC": auc}
        print(f"{name:<25} {acc*100:>9.2f}%  {prec:>10.4f}  {rec:>9.4f}  {f1:>9.4f}  {auc:>10.4f}")


    # ── TASK 2: Hyperparameter Tuning on Random Forest ──────────
    print("\n" + "="*60)
    print("  HYPERPARAMETER TUNING — Random Forest (GridSearchCV)")
    print("="*60)

    # Default RF baseline first
    baseline_rf = RandomForestClassifier(class_weight="balanced", random_state=42, n_jobs=-1)
    baseline_rf.fit(X_train, y_train)
    base_pred = baseline_rf.predict(X_test)
    base_prob = baseline_rf.predict_proba(X_test)[:, 1]
    base_acc  = accuracy_score(y_test, base_pred)
    base_auc  = roc_auc_score(y_test, base_prob)
    print(f"\n[BASELINE] Default RF  →  Accuracy: {base_acc*100:.2f}%  |  ROC-AUC: {base_auc:.4f}")

    # GridSearchCV tuning
    param_grid = {
        "n_estimators"     : [100, 200, 300],
        "max_depth"        : [None, 10, 20],
        "min_samples_split": [2, 5, 10],
        "max_features"     : ["sqrt", "log2"],
        "class_weight"     : ["balanced"],
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    rf = RandomForestClassifier(random_state=42, n_jobs=-1)

    print(f"\n[INFO] Running GridSearchCV — {len(param_grid['n_estimators']) * len(param_grid['max_depth']) * len(param_grid['min_samples_split']) * len(param_grid['max_features'])} configs × 5 folds...")

    grid_search = GridSearchCV(rf, param_grid, cv=cv, scoring="roc_auc", n_jobs=-1, verbose=1)
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_
    print(f"\n[BEST PARAMS]  {grid_search.best_params_}")
    print(f"[BEST CV AUC]  {grid_search.best_score_:.4f}")

    # Tuned model evaluation
    y_pred      = best_model.predict(X_test)
    y_pred_prob = best_model.predict_proba(X_test)[:, 1]
    accuracy    = accuracy_score(y_test, y_pred)
    precision   = precision_score(y_test, y_pred)
    recall      = recall_score(y_test, y_pred)
    f1          = f1_score(y_test, y_pred)
    auc         = roc_auc_score(y_test, y_pred_prob)
    cm          = confusion_matrix(y_test, y_pred)

    # Before vs After comparison
    print("\n" + "="*60)
    print("       BEFORE vs AFTER TUNING")
    print("="*60)
    print(f"{'Metric':<15} {'Default RF':>12} {'Tuned RF':>12} {'Improvement':>14}")
    print("-"*55)
    print(f"{'Accuracy':<15} {base_acc*100:>11.2f}%  {accuracy*100:>11.2f}%  {(accuracy-base_acc)*100:>+12.2f}%")
    print(f"{'ROC-AUC':<15} {base_auc:>12.4f}  {auc:>12.4f}  {(auc-base_auc):>+14.4f}")

    print("\n" + "="*60)
    print("        FINAL TUNED MODEL RESULTS")
    print("="*60)
    print(f"  Accuracy  : {accuracy*100:.2f}%")
    print(f"  Precision : {precision:.4f}")
    print(f"  Recall    : {recall:.4f}")
    print(f"  F1 Score  : {f1:.4f}")
    print(f"  ROC-AUC   : {auc:.4f}")
    print(f"\n  Confusion Matrix:\n{cm}")
    print("\n  Classification Report:")
    print(classification_report(y_test, y_pred, target_names=["No Churn", "Churn"]))
    print("="*60)

    importances = pd.Series(best_model.feature_importances_, index=X.columns).sort_values(ascending=False)
    print("\n[INFO] Top 10 Feature Importances:")
    print(importances.head(10).to_string())

    return best_model, scaler, features

# ── Step 4: Persist artifacts ──────────────────────────────────────────────────
def save_artifacts(model, scaler, label_encoders, feature_columns):
    joblib.dump(model,          MODEL_PATH)
    joblib.dump(scaler,         SCALER_PATH)
    joblib.dump(label_encoders, ENCODER_PATH)
    joblib.dump(feature_columns, FEATURES_PATH)
    print(f"\n[SUCCESS] Artifacts saved to: {ARTIFACTS_DIR}/")
    print(f"          ├── churn_model.pkl")
    print(f"          ├── scaler.pkl")
    print(f"          ├── label_encoders.pkl")
    print(f"          └── feature_columns.pkl")

# ── Main ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    df = fetch_data(DB_PATH)
    df, label_encoders, numerical_cols, categorical_cols = preprocess(df)
    model, scaler, feature_columns = train_and_evaluate(df)
    save_artifacts(model, scaler, label_encoders, feature_columns)