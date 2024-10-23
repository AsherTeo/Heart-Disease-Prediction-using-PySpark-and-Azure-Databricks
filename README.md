# Overview
In this project, I developed a heart failure prediction model using PySpark and Azure Databricks, leveraging MLlib for training and testing. The pipeline includes data loading, preprocessing, feature engineering, model training, fine-tuning, and evaluation.

The Kaggle Heart Failure Prediction dataset was uploaded to Azure Blob Storage. Exploratory Data Analysis (EDA) was conducted by converting the dataset from a Spark DataFrame to a Pandas DataFrame, which facilitated detailed visualizations using Seaborn and Matplotlib 

 A binary classification model was built using PySpark's MLlib, evaluating various machine learning algorithms before selecting the top three for hyperparameter optimization with Hyperopt.
The performance of the selected models was assessed using the F1 score, and the classification threshold was adjusted to enhance the balance between sensitivity and specificity.

This project demonstrates my proficiency in using PySpark and MLlib for machine learning tasks.

# Dataset

The Heart Failure Prediction Dataset from [kaggle](https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction) includes various attributes related to patient demographics and health indicators. Key features of the dataset are as follows:
- Age: Age of the patient (years)
- Sex: Gender of the patient (M: Male, F: Female)
- ChestPainType: Type of chest pain (TA: Typical Angina, ATA: Atypical Angina, NAP: Non-Anginal Pain, ASY: Asymptomatic)
- RestingBP: Resting blood pressure (mm Hg)
- Cholesterol: Serum cholesterol (mm/dl)
- ...
- HeartDisease: Output class indicating heart disease presence (1: heart disease, 0: normal)

# Machine Learning Pipline

## 1. Azure DataBrick Authentication & Azure Blob Storage
- Create a new container in Azure Blob Storage for Heart Failure Prediction Dataset.
- Prepare Azure Storage Access Information such as Container Name, Storage Account Name & SAS Token
- Use PySpark to mount the Azure Blob Storage container in Azure Databricks for easy access to the dataset.

## 2. Data Preprocessing
- **Handle missing values**
- **Remove duplicate values**
- **Analyze the distribution of each feature.**

## 3. Feature Engineering
- **Assess skewness and apply the Box-Cox transformation for features with significant skewness.**
- **Utilize Chi-Square tests for categorical feature selection, retaining features with p-values below 0.05.**
- **Employ ANOVA for numerical feature selection, retaining features with p-values below 0.05.**

## 4. Outliers
- **Plot box plots for numerical features to visualize outliers**
- **Apply the Interquartile Range (IQR) method to remove data points that fall below the 25th percentile or above the 75th percentile.**

## 5. Model Selection
- **Split the data** into training, validation and testing sets
- **StringIndexer**: To convert categorical columns into numerical indices.
- **OneHotEncoder**: To convert the indexed categorical columns into one-hot vectors.
- **StandardScaler**: To normalize numerical features.
- **VectorAssembler**: To assemble all features (both numerical and one-hot encoded) into a single vector.
- **Pipeline**: To combine all transformations in a sequence.
