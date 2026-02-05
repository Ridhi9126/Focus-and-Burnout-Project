import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import joblib

df = pd.read_json("ml/dataset_v1.json")

label_map={
    "Low": 0,
    "Medium": 1,
    "High": 2
    }
reverse_map={
    0:"Low",
    1:"Medium",
    2:"High"
}

df["burnout_label"] = df["burnout_label"].map(label_map)

X = df[['study_hours', 'sleep_hours', 'breaks', 'stress_level']]
y = df['burnout_label']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression(max_iter=1000))
])

pipeline.fit(X_train, y_train)

y_pred = pipeline.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", cm)

coef = pipeline.named_steps["model"].coef_[2]
coef_df = pd.DataFrame({
    "Feature": X.columns,
    "Coefficient": coef
}).sort_values(by="Coefficient", ascending=False)#need to search

print(coef_df)

plt.figure()
plt.scatter(df['study_hours'],df['burnout_label'])
plt.xlabel("study hours")
plt.ylabel("burnout label")
plt.title("study hours vs burnout label")
plt.show()

plt.figure()
plt.scatter(df['sleep_hours'],df['burnout_label'])
plt.xlabel("sleep hours")
plt.ylabel("burnout label")
plt.title("sleep hours vs burnout label")
plt.show()

joblib.dump(pipeline,"ml/burnout_v1.pkl")
joblib.dump(reverse_map, "ml/label_map.pkl")

print("\nModel saved as burnout_v1.pkl")
print("Label map saved as label_map.pkl")