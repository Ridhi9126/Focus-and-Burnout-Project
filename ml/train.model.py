import pandas as pd


df=pd.read_json("ml/dataset.json")
print(df.head())
print(df.columns)
print(df.shape)

# -------------------------------
# Dataset Explanation
# -------------------------------
# Features (X):
# study_hours   -> number of hours spent studying
# sleep_hours   -> number of hours of sleep
# breaks        -> number of breaks taken
# stress_level  -> self-reported stress level

# Label (y):
# burnout_label -> burnout category (Low, Medium, High)

df["burnout_label"]=df["burnout_label"].map({
    "low":0,
    "medium":1,
    "high":2
})

X = df[['study_hours', 'sleep_hours', 'breaks', 'stress_level']]
y = df['burnout_label']

print(df["burnout_label"].head())