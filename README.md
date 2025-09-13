Explainable AI based Intrusion Detection System (IDS)
Project Overview

This project focuses on detecting Denial of Service (DoS) attacks using Machine Learning models and making the predictions explainable with XAI (SHAP).
Traditional IDS models act as black boxes and only output “attack detected” without explaining the reason.
Our approach provides high accuracy detection while also showing why the model flagged traffic as malicious.

 Problem Statement

DoS and DDoS attacks overwhelm systems by sending a huge volume of packets.
Existing IDS solutions achieve good accuracy but lack explainability, making them less trustworthy for analysts.

🛠️ Technologies Used

Python 3

Scikit-learn – ML models (Logistic Regression, SVM, Random Forest)

XGBoost – Gradient boosting model for highest accuracy

SHAP – Explainable AI framework

Matplotlib / Seaborn – Visualizations

Methodology

Workflow:

Input Traffic →

Preprocessing →

ML Models (LogReg, SVM, Random Forest, XGBoost) →

SHAP Explanations →

Analyst Decision

*Results*
Model	Accuracy	Precision	Recall	F1-score
Logistic Regression	91%	90%	89%	89%
SVM	                94%	93%	92%	92%
Random Forest     	97%	96%	97%	96%
XGBoost	            98%	97%	98%	97%

Confusion Matrix: XGBoost detected 50,000 attacks correctly with only 3 misses.

SHAP Results: Key features influencing detection were:

Flow Duration → abnormal sessions

Packet Rate → strong sign of flooding

Failed Logins → brute-force attempts

 Role of XAI (SHAP)

Makes IDS transparent & trustworthy.

Explains why a connection was flagged as an attack.

Helps analysts in root cause analysis and decision making.

 Future Scope

Extend to other attack types (R2L, U2R, Probe).

Integrate with real-time network monitoring.

Use deep learning + XAI for more complex attacks.

How to Run
# Install dependencies
pip install -r requirements.txt

# Preprocess dataset
python src/preprocessing.py

# Train models
python src/models.py

# Evaluate models
python src/evaluation.py

# Run SHAP explanations
python src/shap_explain.py
