#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 29 20:13:32 2025

@author: hannah
"""

import pandas as pd
import matplotlib.pyplot as plt
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_classif

from imblearn.combine import SMOTEENN

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    roc_auc_score,
)

# ==============================================================
print("\n================= 1. LOAD CLEANED DATA =================")
df = pd.read_csv("/Users/hannadelala/Library/Mobile Documents/com~apple~CloudDocs/2-Study-Materials/Centennial-College/Semester-6/COMP309-Data-Warehouse/Bicycle_theft_Group_Group4_section_001#COMP309Project/Bicycle_Thefts_Open_Data_cleaned.csv")
print("Cleaned dataset shape:", df.shape)

# ==============================================================
print("\n================= 2. HANDLE TARGET VARIABLE =============")
# Keep only STOLEN / RECOVERED
df = df[df["STATUS"].isin(["STOLEN", "RECOVERED"])]

# Map to binary: 1 = STOLEN, 0 = RECOVERED
df["STATUS"] = df["STATUS"].map({"RECOVERED": 0, "STOLEN": 1})
print("After filtering STATUS:", df.shape)
print("Class proportion:\n", df["STATUS"].value_counts(normalize=True))

# ==============================================================
print("\n=========== 3. DROP HIGH-CARDINALITY + UNUSED COLS ======")
drop_high_cardinality = ["BIKE_MODEL", "BIKE_MAKE", "BIKE_COLOUR"]
drop_cols = ["EVENT_UNIQUE_ID", "OBJECTID", "x", "y"]
drop_all = drop_high_cardinality + drop_cols

df = df.drop(columns=[c for c in drop_all if c in df.columns])
print("Shape after dropping:", df.shape)

# ==============================================================
print("\n================= 4. TRAIN / TEST SPLIT =================")
X = df.drop("STATUS", axis=1)
y = df["STATUS"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42
)

print("Train:", X_train.shape)
print("Test:", X_test.shape)
print("Train class proportion:\n", y_train.value_counts(normalize=True))

# ==============================================================
print("\n================= 5. DEFINE PIPELINES ===================")
categorical_cols = X_train.select_dtypes(include=["object"]).columns
numeric_cols = X_train.select_dtypes(include=["int64", "float64"]).columns

print("Categorical:", list(categorical_cols))
print("Numeric:", list(numeric_cols))

numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", MinMaxScaler())
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("num", numeric_pipeline, numeric_cols),
    ("cat", categorical_pipeline, categorical_cols)
])

# ==============================================================
print("\n=========== 6. FIT PREPROCESSOR + FEATURE SELECTION =====")
X_train_processed = preprocessor.fit_transform(X_train)
print("X_train after preprocessing:", X_train_processed.shape)

k_features = 15
selector = SelectKBest(score_func=f_classif, k=k_features)
X_train_selected = selector.fit_transform(X_train_processed, y_train)
print(f"X_train after SelectKBest (k={k_features}):", X_train_selected.shape)


# Some transformers return sparse matrices. convert to dense if needed
try:
    X_train_selected_dense = X_train_selected.toarray()
except AttributeError:
    X_train_selected_dense = X_train_selected

# ==============================================================
print("\n=========== 7. TRANSFORM TEST USING TRAINED PIPELINE ===")
X_test_processed = preprocessor.transform(X_test)
X_test_selected = selector.transform(X_test_processed)
try:
    X_test_selected_dense = X_test_selected.toarray()
except AttributeError:
    X_test_selected_dense = X_test_selected

print("X_test after preprocessing + SelectKBest:", X_test_selected_dense.shape)

# ==============================================================
print("\n=========== 8. HANDLE IMBALANCE WITH SMOTEENN ===========")
print("Before SMOTEENN:\n", y_train.value_counts(normalize=True))

smote_enn = SMOTEENN(random_state=42)
X_train_resampled, y_train_resampled = smote_enn.fit_resample(
    X_train_selected_dense, y_train
)

print("After SMOTEENN:", X_train_resampled.shape)
print("Balanced target:\n", y_train_resampled.value_counts(normalize=True))

print("\n================= DATA MODELLING COMPLETE =================")
print("Final X_train:", X_train_resampled.shape)
print("Final X_test:", X_test_selected_dense.shape)

# ==================================================================
#                  PREDICTIVE MODEL BUILDING
# ==================================================================

# ============================ 9. LOGISTIC REGRESSION ============================

log_model = LogisticRegression(max_iter=1000)
log_model.fit(X_train_resampled, y_train_resampled)

log_preds = log_model.predict(X_test_selected_dense)
log_probs = log_model.predict_proba(X_test_selected_dense)[:, 1]

log_acc = accuracy_score(y_test, log_preds)
log_cm = confusion_matrix(y_test, log_preds)
log_auc = roc_auc_score(y_test, log_probs)

print("\n--- LOGISTIC REGRESSION RESULTS ---")
print(f"Accuracy: {log_acc:.3f}")
print(f"AUC:      {log_auc:.3f}")
print("Confusion Matrix:\n", log_cm)

ConfusionMatrixDisplay(log_cm).plot()
plt.title("Logistic Regression Confusion Matrix")
plt.show()

# ============================ 10. DECISION TREE ================================

tree_model = DecisionTreeClassifier(max_depth=6, random_state=42)
tree_model.fit(X_train_resampled, y_train_resampled)

tree_preds = tree_model.predict(X_test_selected_dense)
tree_probs = tree_model.predict_proba(X_test_selected_dense)[:, 1]

tree_acc = accuracy_score(y_test, tree_preds)
tree_cm = confusion_matrix(y_test, tree_preds)
tree_auc = roc_auc_score(y_test, tree_probs)

print("\n--- DECISION TREE RESULTS ---")
print(f"Accuracy: {tree_acc:.3f}")
print(f"AUC:      {tree_auc:.3f}")
print("Confusion Matrix:\n", tree_cm)

ConfusionMatrixDisplay(tree_cm).plot()
plt.title("Decision Tree Confusion Matrix")
plt.show()

# ======================= 11. ROC CURVE COMPARISON ==============================

fpr1, tpr1, _ = roc_curve(y_test, log_probs)
fpr2, tpr2, _ = roc_curve(y_test, tree_probs)

plt.plot(fpr1, tpr1, label=f"Logistic Regression (AUC = {log_auc:.3f})")
plt.plot(fpr2, tpr2, label=f"Decision Tree (AUC = {tree_auc:.3f})")
plt.plot([0, 1], [0, 1], linestyle="--")

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve Comparison")
plt.legend()
plt.show()

# ============================ 12. BEST MODEL SELECTION ==========================

if log_auc > tree_auc:
    best_model = "Logistic Regression"
else:
    best_model = "Decision Tree"

print("\nBEST PERFORMING MODEL:", best_model)

# ============================ 13. SAVE BEST MODEL BUNDLE =======================

# Choose actual best classifier object
if best_model == "Logistic Regression":
    best_clf = log_model
else:
    best_clf = tree_model

# Create a bundle with everything the API needs
model_bundle = {
    "preprocessor": preprocessor,
    "selector": selector,
    "classifier": best_clf,
}

# Serialize with pickle
with open("bike_return_model.pkl", "wb") as f:
    pickle.dump(model_bundle, f)

print("Model bundle saved to bike_return_model.pkl")
