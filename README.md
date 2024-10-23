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

## 3a. Feature Engineering
- **Assess skewness and apply the Box-Cox transformation for features with significant skewness.**
- **Utilize Chi-Square tests for categorical feature selection, retaining features with p-values below 0.05.**
- **Employ ANOVA for numerical feature selection, retaining features with p-values below 0.05.**

## 3b. Outliers
- **Plot box plots for numerical features to visualize outliers**
- **Apply the Interquartile Range (IQR) method to remove data points that fall below the 25th percentile or above the 75th percentile.**

## 4a. Machine Learning Pipeline
- **StringIndexer**: To convert categorical columns into numerical indices.
- **OneHotEncoder**: To convert the indexed categorical columns into one-hot vectors.
- **StandardScaler**: To normalize numerical features.
- **VectorAssembler**: To assemble all features (both numerical and one-hot encoded) into a single vector.
- **Pipeline**: To combine all transformations in a sequence.
  
## 4b. Model Selection
- **Split the data into training, validation, and test sets.**
- **Train various models (e.g., XGBoost, LightGBM, Logistic Regression) on the training set and validate using the validation set.**
- **Select the top 3 models for fine tunning**

### Top 3 Validation Results

| **Baseline Model** | **F1 Score** |
|-------------------|--------------|
| Random Forest      | 0.941748     |
| LightGBM          | 0.922403     |
| XGBoost           | 0.902968     |

## 5. Model FineTuning 
- **Tuning method: HyperOpt was used to fine-tune hyperparameters for each model.**
- **Best models: The table below shows the performance of top models after fine-tuning.**
| **Model**       | **F1 Score** | **Precision**   | **Recall** |
|-------------|-----------|----------|----------|
| LightGBM    | 0.951484  | 0.951691 | 0.951456 | 
| XGBoost         | 0.941803  | 0.942568 | 0.941748 | 
| Random Forest    | 0.941748  | 0.941748 | 0.941748 | 


