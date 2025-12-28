import pandas as pd
from sklearn.model_selection import train_test_split
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
    "Low":0,
    "Medium":1,
    "High":2
})

x = df[['study_hours', 'sleep_hours', 'breaks', 'stress_level']]
y = df['burnout_label']

print(df["burnout_label"].head())

x_train, x_test, y_train, y_test= train_test_split(x,y,test_size=0.2,random_state=42)
print("training data size",x_train.shape)
print("testing data size",x_test.shape)