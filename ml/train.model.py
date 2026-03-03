# ==========================================================
# TRAIN_MODEL.PY
# Burnout Prediction ML Pipeline using Logistic Regression
# ==========================================================

# -------------------------------
# IMPORT REQUIRED LIBRARIES
# -------------------------------

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import joblib


# -------------------------------
# DATA LOADING
# -------------------------------
# Load dataset from JSON file
# This dataset contains:
# study_hours, sleep_hours, breaks, stress_level, burnout_label

df = pd.read_json("ml/dataset_v1.json")


# -------------------------------
# LABEL ENCODING
# -------------------------------
# Convert categorical labels into numerical format

label_map = {
    "Low": 0,
    "Medium": 1,
    "High": 2
}

# Reverse map used later for prediction decoding
reverse_map = {
    0: "Low",
    1: "Medium",
    2: "High"
}

# Apply mapping
df["burnout_label"] = df["burnout_label"].map(label_map)


# -------------------------------
# FEATURE SELECTION
# -------------------------------
# X → Input Features
# y → Target Label

X = df[['study_hours', 'sleep_hours', 'breaks', 'stress_level']]
y = df['burnout_label']


# -------------------------------
# TRAIN TEST SPLIT
# -------------------------------
# 80% Training
# 20% Testing

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# -------------------------------
# CREATE ML PIPELINE
# -------------------------------
# Pipeline steps:
# Step 1 → Standardize data
# Step 2 → Apply Logistic Regression model

pipeline = Pipeline([

    ("scaler", StandardScaler()),

    ("model", LogisticRegression(max_iter=1000))

])


# -------------------------------
# MODEL TRAINING
# -------------------------------

pipeline.fit(X_train, y_train)


# -------------------------------
# MODEL EVALUATION
# -------------------------------

y_pred = pipeline.predict(X_test)

print("================================")
print("MODEL PERFORMANCE")
print("================================")

print("Accuracy:", accuracy_score(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix:")
print(cm)


# -------------------------------
# FEATURE IMPORTANCE
# -------------------------------
# Shows which feature affects burnout most

coef = pipeline.named_steps["model"].coef_[2]

coef_df = pd.DataFrame({

    "Feature": X.columns,
    "Coefficient": coef

}).sort_values(by="Coefficient", ascending=False)

print("\nFeature Importance:")
print(coef_df)


# -------------------------------
# DATA VISUALIZATION
# -------------------------------

# Study Hours vs Burnout

plt.figure()

plt.scatter(df['study_hours'], df['burnout_label'])

plt.xlabel("Study Hours")
plt.ylabel("Burnout Label")

plt.title("Study Hours vs Burnout Level")

plt.show()


# Sleep Hours vs Burnout

plt.figure()

plt.scatter(df['sleep_hours'], df['burnout_label'])

plt.xlabel("Sleep Hours")
plt.ylabel("Burnout Label")

plt.title("Sleep Hours vs Burnout Level")

plt.show()


# -------------------------------
# SAVE TRAINED MODEL
# -------------------------------

joblib.dump(pipeline, "ml/burnout_v1.pkl")

joblib.dump(reverse_map, "ml/label_map.pkl")


print("\n================================")
print("MODEL SAVED SUCCESSFULLY")
print("================================")

print("Model file → burnout_v1.pkl")

print("Label map → label_map.pkl")
