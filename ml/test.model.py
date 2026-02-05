import joblib
import pandas as pd

# Load model and label map
model = joblib.load("ml/burnout_v1.pkl")
label_map = joblib.load("ml/label_map.pkl")


def predict_burnout(study, sleep, breaks, stress):

    # Input validation
    if study < 0 or sleep < 0 or breaks < 0:
        return "Invalid Input: Values cannot be negative"

    if stress < 0 or stress > 10:
        return "Invalid Input: Stress must be between 0 and 10"

    # Prepare input
    data = pd.DataFrame([{
        "study_hours": study,
        "sleep_hours": sleep,
        "breaks": breaks,
        "stress_level": stress
    }])

    # Predict
    pred = model.predict(data)

    # Convert to label
    result = label_map[pred[0]]

    return result


# Test prediction
if __name__ == "__main__":

    study_hours = 6
    sleep_hours = 5
    breaks = 2
    stress_level = 7

    result = predict_burnout(
        study_hours,
        sleep_hours,
        breaks,
        stress_level
    )

    print("Input Data:")
    print("Study Hours:", study_hours)
    print("Sleep Hours:", sleep_hours)
    print("Breaks:", breaks)
    print("Stress Level:", stress_level)

    print("\nPredicted Burnout Level:", result)
