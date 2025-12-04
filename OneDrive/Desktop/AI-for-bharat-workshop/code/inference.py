import joblib
import pandas as pd

def predict(input_data):
    model = joblib.load("model.pkl")
    df = pd.DataFrame([input_data])
    prediction = model.predict(df)[0]
    return prediction

if __name__ == "__main__":
    sample = {"feature1": 5, "feature2": 3, "feature3": 1}
    result = predict(sample)
    print("Prediction:", result)
