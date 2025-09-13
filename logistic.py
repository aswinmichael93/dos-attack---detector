import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import shap
import matplotlib.pyplot as plt


df = pd.read_csv('data/Wednesday-workingHours.pcap_ISCX.csv')
print("Loaded:", df.shape)


df.replace([float('inf'), -float('inf')], pd.NA, inplace=True)
df.dropna(inplace=True)


df['Attack'] = df[' Label'].apply(lambda x: 0 if x == 'BENIGN' else 1)


df.drop(columns=['Flow ID', 'Source IP', 'Destination IP', 'Timestamp'], inplace=True, errors='ignore')


le = LabelEncoder()
categorical_cols = df.select_dtypes(include=['object']).columns
for col in categorical_cols:
    if col != ' Label':
        df[col] = le.fit_transform(df[col])

print("Preprocessing complete. Shape:", df.shape)


X = df.drop(columns=[' Label', 'Attack'])
y = df['Attack']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Train size:", X_train.shape[0], " Test size:", X_test.shape[0])


model = LogisticRegression(max_iter=500, solver='lbfgs')
model.fit(X_train, y_train)


y_pred = model.predict(X_test)
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nGenerating SHAP summary plot for Logistic Regression...")

X_sample = shap.sample(X_test, 50, random_state=42)

explainer = shap.KernelExplainer(model.predict_proba, shap.sample(X_train, 50, random_state=42))
shap_values = explainer.shap_values(X_sample)


if isinstance(shap_values, list):
    shap.summary_plot(shap_values[1], X_sample)  
else:
    shap.summary_plot(shap_values, X_sample)     
