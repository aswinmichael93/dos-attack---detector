import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from xgboost import XGBClassifier
import shap
import matplotlib.pyplot as plt

df = pd.read_csv('data/Wednesday-workingHours.pcap_ISCX.csv')
print(" Dataset Loaded:", df.shape)
print(" Columns:", df.columns.tolist())

df.replace([float('inf'), -float('inf')], pd.NA, inplace=True)
df.dropna(inplace=True)

df['Attack'] = df[' Label'].apply(lambda x: 0 if x == 'BENIGN' else 1)
df.drop(columns=['Flow ID', 'Source IP', 'Destination IP', 'Timestamp'], inplace=True, errors='ignore')

le = LabelEncoder()
categorical_cols = df.select_dtypes(include=['object']).columns
for col in categorical_cols:
    if col != ' Label':
        df[col] = le.fit_transform(df[col])

print(" Preprocessing Complete. Shape:", df.shape)

X = df.drop(columns=[' Label', 'Attack'])
y = df['Attack']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("Train size:", X_train.shape[0], " Test size:", X_test.shape[0])

model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("\n Classification Report:")
print(classification_report(y_test, y_pred))

print("\n Confusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap="Blues")
plt.title("Confusion Matrix - XGBoost IDS")
plt.show()

print("\nGenerating SHAP summary plot...")
explainer = shap.Explainer(model, X_train)
X_sample = X_test.sample(100, random_state=42)
shap_values = explainer(X_sample)
shap.summary_plot(shap_values, X_sample)
