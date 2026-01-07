# Student Burnout Prediction using Machine Learning

## Project Overview

This project is a Machine Learning based system that predicts the **burnout level of a student** (Low, Medium, High) based on lifestyle and academic factors such as:

- Study hours
- Sleep hours
- Number of breaks
- Stress level

The goal of this project is to demonstrate how **supervised machine learning** can be used to learn patterns from data and make predictions about student well-being.

---

## Approach

- Since real-world labeled burnout data was not easily available, a **synthetic dataset** was created based on reasonable assumptions about student lifestyle and stress.
- The dataset contains multiple samples with features like study hours, sleep hours, breaks, and stress level, along with a burnout label.
- A **Machine Learning classification model** (Logistic Regression) is trained on this data.
- The model learns the relationship between input features and burnout level automatically using the `.fit()` method.

> No hard-coded rules are used during prediction. The model learns patterns purely from data.

---

## Dataset

Each data sample contains:

- `study_hours` → Number of hours studied per day  
- `sleep_hours` → Number of hours slept per day  
- `breaks` → Number of breaks taken  
- `stress_level` → Stress level on a scale (e.g., 1–10)  
- `burnout_label` → Low / Medium / High  

The dataset was manually designed to simulate realistic student behavior.

---

## Model Training

Steps:
1. Load dataset from JSON file
2. Split into training and testing sets
3. Apply preprocessing (scaling using pipeline)
4. Train the model using Logistic Regression
5. Evaluate using accuracy score and confusion matrix

---

## Results & Improvement

- Initial model accuracy: **0.8**
- After improving dataset quality and balancing samples: **0.9**

This shows that **better data quality directly improves model performance**.

---

## Evaluation Metrics

- Accuracy Score
- Confusion Matrix

---

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib

---

## How to Run

1. Clone the repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
