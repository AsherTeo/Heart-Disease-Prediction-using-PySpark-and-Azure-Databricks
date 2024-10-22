# Overview
In this project, I developed a heart disease prediction model using PySpark and Azure Databricks, leveraging MLlib for training and testing. The project includes building an end-to-end machine learning pipeline that covers data loading, preprocessing, feature engineering, model training, fine-tuning, and evaluation.

The dataset, sourced from Kaggle's Heart Disease Prediction, was uploaded to Azure Blob Storage for storage. For exploratory data analysis (EDA), the data was converted from a Spark DataFrame to a Pandas DataFrame, allowing for detailed visualizations using Seaborn and Matplotlib.

Using PySpark's MLlib, I applied various machine learning algorithms to develop a binary classification model. The model was fine-tuned with Hyperopt for hyperparameter optimization and evaluated using  F1 score metrics to gauge its effectiveness in predicting heart disease. Additionally, the classification threshold was adjusted to further enhance the model’s performance, improving its balance between sensitivity and specificity.

This project demonstrates a practical application of PySpark and Databricks for large-scale data processing and machine learning model development, from data ingestion to model evaluation.
