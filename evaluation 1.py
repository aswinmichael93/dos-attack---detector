import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

data = pd.read_csv("data/clean_dataset.csv")

X = data.drop("label", axis=1)
y = data["label"]


log_reg = joblib.load("results/Logistic_Regression.pkl")
rf = joblib.load("results/Random_Forest.pkl")


y_pred_logreg = log_reg.predict(X)
y_pred_rf = rf.predict(X)


def evaluate_model(name, y_true, y_pred):
    print(f"\n{name} Evaluation:")
    print("Accuracy :", accuracy_score(y_true, y_pred))
    print("Precision:", precision_score(y_true, y_pred))
    print("Recall   :", recall_score(y_true, y_pred))
    print("F1 Score :", f1_score(y_true, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_true, y_pred))


evaluate_model("Logistic Regression", y, y_pred_logreg)
evaluate_model("Random Forest", y, y_pred_rf)
