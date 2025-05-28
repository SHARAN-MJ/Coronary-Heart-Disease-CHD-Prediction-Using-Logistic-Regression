### Coronary Heart Disease (CHD) Prediction Using Logistic Regression
# Project Overview
This project implements a robust machine learning pipeline to **predict the presence of Coronary Heart Disease (CHD)** based on clinical and lifestyle parameters. It leverages logistic regression with hyperparameter tuning and advanced techniques to handle imbalanced data, ensuring reliable and interpretable predictions.

# Key Features
**Comprehensive Data Preprocessing:** Cleans and encodes categorical features, handles missing values.

**Feature Scaling:** Applies StandardScaler for normalized inputs to the model.

**Imbalanced Data Handling:** Uses SMOTE (Synthetic Minority Over-sampling Technique) to balance the dataset.

**Hyperparameter Tuning:** Employs GridSearchCV to optimize Logistic Regression parameters.

**Model Evaluation:** Reports accuracy, detailed classification metrics, and displays a confusion matrix heatmap.

**Model Persistence:** Saves trained model and scaler for future prediction without retraining.

**Interactive Manual Input Mode:** Predict CHD based on user-provided data when the dataset is unavailable.

# Dataset
The model is trained on a dataset (CHDdata.csv) with the following columns:

![image](https://github.com/user-attachments/assets/795919bd-e3a7-405d-8afd-705b72fdaef4)


# How to Use
**1. Training & Evaluation (with Dataset)**
Place CHDdata.csv in the project directory.

Run the script:

![image](https://github.com/user-attachments/assets/6807153b-a1e4-4f18-97ad-146dd0dbedea)

The model will train, tune, and display evaluation results.

Trained model and scaler will be saved as chd_logistic_model.pkl and scaler.pkl.

**2. Manual Prediction Mode (without Dataset)**
If no dataset is found, the script will prompt for health parameters input.

Enter values as requested.

The saved model and scaler will be used to predict CHD presence and probability.

## Example Workflow
*bash*

$ python chd_prediction.py
✅ Dataset found. Proceeding with training...

✅ Logistic Regression Accuracy: 0.8542

📋 Classification Report:
               precision    recall  f1-score   support
           0       0.86      0.89      0.88       100
           1       0.83      0.79      0.81        80
    accuracy                           0.85       180
   macro avg       0.85      0.84      0.84       180
weighted avg       0.85      0.85      0.85       180

✅ Model and scaler saved for future use.
Or in manual mode:

*yaml*

📝 Please enter the following values for prediction:
Systolic Blood Pressure (sbp): 140
Tobacco usage (tobacco): 3.5
LDL cholesterol (ldl): 130
Adiposity: 24.5
Family history (Present or Absent): Present
Type A behavior score (typea): 45
Obesity index: 30
Alcohol consumption: 12
Age: 54

🔍 Prediction: CHD Present
📊 Probability of CHD: 78.45%


## Dependencies
Python 3.x

pandas

numpy

scikit-learn

imbalanced-learn

matplotlib

seaborn

joblib

Install via pip:

*bash*

pip install pandas numpy scikit-learn imbalanced-learn matplotlib seaborn joblib

📚 Project Structure
bash


├── CHDdata.csv                # Dataset file (CSV)
├── chd_prediction.py          # Main Python script containing model & prediction code
├── chd_logistic_model.pkl     # Saved Logistic Regression model (post-training)
├── scaler.pkl                 # Saved StandardScaler (post-training)
└── README.md                  # Project documentation

# 🧑‍💻 Author
SHARAN MJ
Feel free to reach out: sharanmaran1349@gmail.com
