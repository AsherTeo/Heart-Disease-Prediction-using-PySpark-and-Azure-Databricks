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
- **Utilize PySpark to identify any missing values and duplicate records in the dataset.**
- **Analyze the distribution of each feature.**
- **Check for Target Imbalance**
- **Remove Irrelevant Features like IDs or names**

## 3. Exploratory Data Analysis (EDA)

This section presents a simple exploratory data analysis (EDA) to understand the characteristics of the heart failure dataset better. Below are two visualizations that provide insights into the data:

- **Left Image: The visualization indicates that men are approximately three times more likely to develop heart disease compared to women.**
- **Right Image: This visualization suggests that individuals experiencing asymptomatic chest pain (Chest Pain ASY) have a significantly higher likelihood of developing heart disease.**
  
  <table>
<tr>
    <td><img src="https://github.com/user-attachments/assets/893fe7fc-2a3a-4abc-826f-acf083ffebb8" alt="Image 1" width="400"/></td>
    <td><img src="https://github.com/user-attachments/assets/83f74d31-4b56-488d-af5e-08bdc33bb3d5" alt="Image 2" width="400"/></td>
</tr>
</table>

## 4. Feature Engineering

- **Assess skewness and apply the Box-Cox transformation for features with significant skewness.**
- **Utilize Chi-Square tests for categorical feature selection, retaining features with p-values below 0.05.**
- **Employ ANOVA for numerical feature selection, retaining features with p-values below 0.05.**
  
  <table>
<tr>
    <td><img src="https://github.com/user-attachments/assets/7a3c3cd1-a6ce-4434-b90b-7a0de2f1ecea" alt="Image 1" width="400"/></td>
    <td><img src="https://github.com/user-attachments/assets/acaf3f3c-2bc4-459a-8745-f1741c9c891e" alt="Image 2" width="400"/></td>
</tr>
</table>

## 5. Outliers
- **Plot box plots for numerical features to visualize outliers**
- **Apply the Interquartile Range (IQR) method to remove data points that fall below the 25th percentile or above the 75th percentile.**
  
<table>
<tr>
    <td><img src="https://github.com/user-attachments/assets/b9abab72-8127-4581-9272-e68b4a70c3df" alt="Image 1" width="400"/></td>
    <td><img src="https://github.com/user-attachments/assets/964d2e84-6487-404d-b60c-1545118f773d" alt="Image 2" width="400"/></td>
</tr>
</table>

## 6. Machine Learning Pipeline
- **StringIndexer**: To convert categorical columns into numerical indices.
- **OneHotEncoder**: To convert the indexed categorical columns into one-hot vectors.
- **StandardScaler**: To normalize numerical features.
- **VectorAssembler**: To assemble all features (both numerical and one-hot encoded) into a single vector.
- **Pipeline**: To combine all transformations in a sequence.
- 
The image below shows an example of the final features after the pipeline processing:

![image](https://github.com/user-attachments/assets/5a570282-b285-466a-a05a-91ac21b50b5d)

## 7. Model Selection
- **Split the data into training, validation, and test sets.**
- **Train various models (e.g., XGBoost, LightGBM, Logistic Regression) on the training set and validate using the validation set.**
- **Select the top 3 models for fine tunning**

### Top 3 Validation Results

| **Baseline Model** | **F1 Score** |
|-------------------|--------------|
| Random Forest      | 0.941748     |
| LightGBM          | 0.922403     |
| XGBoost           | 0.902968     |

## 8. Model FineTuning 
- **Tuning method: HyperOpt was used to fine-tune hyperparameters for each model.**
- **Best models: The table below shows the performance of top models after fine-tuning.**
  
| **Model**         | **F1 Score** | **Precision** | **Recall**   |
|-------------------|--------------|---------------|--------------|
| LightGBM          | 0.951484     | 0.951691      | 0.951456     |
| XGBoost           | 0.941803     | 0.942568      | 0.941748     |
| Random Forest      | 0.941748     | 0.941748      | 0.941748     |

## 9. Evaluation 
- **Evaluate top models on the test dataset using metrics such as F1-score, precision, and recall.**

| **Model**         | **F1 Score** | **Precision** | **Recall**   |
|-------------------|--------------|---------------|--------------|
| LightGBM          | 0.901270     | 0.904431      | 0.901786     |
| XGBoost           | 0.900949     | 0.907403      | 0.901786     |
| Random Forest      | 0.900949     | 0.907403      | 0.901786     |

The best model for heart failure prediction is the fine-tuned LightGBM.

## 10. Classification Threshold
- **Adjusting the classification threshold is another method for fine-tuning model performance.**
- **The optimal threshold is determined by maximizing the difference between TPR and FPR, which is calculated using the ROC AUC score.**
  
![image](https://github.com/user-attachments/assets/669d9f9c-e884-42ca-bb70-955c13579b69)

## Limitations:
- Cross-validation is generally preferred for generalizing data, but due to limitations in the trial version, performing cross-validation with HyperOpt is slow, and I have limited credits for Azure free trials. A more robust approach would involve using cross-validation in conjunction with fine-tuning.
- Another limitation is the inability to clearly demonstrate feature importance, which is crucial for explaining to non-technical stakeholders which features are most influential in predicting heart failure. This lack of transparency can hinder communication of model insights to a broader audience.

## Environment
- Machine Learning Framework: Apache Spark 3.5.0
- Programming Language: Scala 2.12
- Virtual Machine Configuration:
   - Type: Standard_DS3_v2
   - Memory: 14 GB
   - Cores: 4
