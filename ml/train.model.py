import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score 

df=pd.read_json("ml/dataset.json")

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
    "Low":0,                        # machine learining model requires numerial labels
    "Medium":1,
    "High":2
})

x = df[['study_hours', 'sleep_hours', 'breaks', 'stress_level']]     # features
y = df['burnout_label']                                              # labels

x_train, x_test, y_train, y_test= train_test_split(x,y,test_size=0.2,random_state=42)    #train-test-split
model = LogisticRegression(max_iter=100)         # max_iter uses the maximum number of iteraions he model can use to converge during training
model.fit(x_train,y_train)

y_pred=model.predict(x_test)

accuracy=accuracy_score(y_test,y_pred)
print("accuracy:",accuracy)

for actual, predicted in zip(y_test,y_pred):        # zip pairs each actual label with its corresponding predicted label so they can be compared together.
    print(f"actual:{actual}, predicted:{predicted}") 