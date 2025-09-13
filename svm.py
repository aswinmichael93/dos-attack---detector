import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
import shap
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("data/Wednesday-workingHours.pcap_ISCX.csv")
df.columns = df.columns.str.strip()

# Preprocess
df.replace([float('inf'), -float('inf')], pd.NA, inplace=True)
df.dropna(inplace=True)
df['Attack'] = df['Label'].apply(lambda x: 0 if x == 'BENIGN' else 1)
df.drop(columns=['Flow ID', 'Source IP', 'Destination IP', 'Timestamp'], inplace=True, errors='ignore')

# Encode categorical columns
le = LabelEncoder()
for col in df.select_dtypes(include=['object']).columns:
    if col != 'Label':
        df[col] = le.fit_transform(df[col])

# Split features and target
X = df.drop(columns=['Label', 'Attack'])
y = df['Attack']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train SVM
model = SVC(kernel="rbf", probability=True, random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print("SVM Results")
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))

# SHAP Explainability
print("\nGenerating SHAP summary plot for SVM...")
X_train_sample = X_train.sample(50, random_state=42)
X_test_sample = X_test.sample(50, random_state=42)

explainer = shap.KernelExplainer(model.predict_proba, X_train_sample)
shap_values = explainer.shap_values(X_test_sample)

if isinstance(shap_values, list):
    shap.summary_plot(shap_values[1], X_test_sample)  # attack class
else:
    shap.summary_plot(shap_values, X_test_sample)
