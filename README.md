# Overview
In this project, I developed a heart failure prediction model using PySpark and Azure Databricks, leveraging MLlib for training and testing. The pipeline includes data loading, preprocessing, feature engineering, model training, fine-tuning, and evaluation.

The Kaggle Heart Failure Prediction dataset was uploaded to Azure Blob Storage. Exploratory Data Analysis (EDA) was conducted by converting the dataset from a Spark DataFrame to a Pandas DataFrame, which facilitated detailed visualizations using Seaborn and Matplotlib 

A binary classification model was built using PySpark's MLlib, followed by fine-tuning with Hyperopt to optimize hyperparameters. The model's performance was evaluated using the F1 score, and the classification threshold was adjusted to enhance the balance between sensitivity and specificity.

This project demonstrates my proficiency in using PySpark and MLlib for machine learning tasks.

# Dataset

The Heart Failure Prediction Dataset from [kaggle](https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction) includes various attributes related to patient demographics and health indicators. Key features of the dataset are as follows:
- Age: Age of the patient (years)
- Sex: Gender of the patient (M: Male, F: Female)
ChestPainType: Type of chest pain (TA: Typical Angina, ATA: Atypical Angina, NAP: Non-Anginal Pain, ASY: Asymptomatic)
RestingBP: Resting blood pressure (mm Hg)
Cholesterol: Serum cholesterol (mm/dl)
FastingBS: Fasting blood sugar (1: if FastingBS > 120 mg/dl, 0: otherwise)
RestingECG: Resting electrocardiogram results (Normal, ST: ST-T wave abnormality, LVH: Left ventricular hypertrophy)
MaxHR: Maximum heart rate achieved (numeric value between 60 and 202)
ExerciseAngina: Exercise-induced angina (Y: Yes, N: No)
Oldpeak: ST depression measured in depression (numeric value)
ST_Slope: Slope of the peak exercise ST segment (Up: upsloping, Flat: flat, Down: downsloping)
HeartDisease: Output class indicating heart disease presence (1: heart disease, 0: normal)
