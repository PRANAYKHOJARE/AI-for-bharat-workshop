import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib

def train_model():
    df = pd.read_csv("data/processed/clean_data.csv")

    X = df.drop("target", axis=1)
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    model = LogisticRegression()
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)

    print("Model Accuracy:", acc)

    joblib.dump(model, "model.pkl")
    print("Model saved to model.pkl")

if __name__ == "__main__":
    train_model()
