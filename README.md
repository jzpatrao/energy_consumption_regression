* Predictive Regression Model for Energy Consumption and CO2 Emission in Seattle Residential Buildings**

1. Introduction:
   - This project aims to build a predictive regression model for energy consumption and CO2 emission in residential buildings located in the city of Seattle.
   - The goal is to develop a model that accurately predicts these variables and gain insights into the factors influencing energy consumption and CO2 emission in residential buildings.

2. Dataset Exploration:
   - The project begins with an exploration of the dataset, examining its structure, features, and any missing values.
   - Abnormal entries are identified and cleaned, and outliers are examined and treated based on their nature.

3. Target Variable Analysis:
   - The distribution of the target variables (energy consumption and CO2 emission) is studied to assess their skewness.
   - If necessary, a log transformation is applied to the target variables to correct skewness and improve the model's performance.

4. Model Selection:
   - Four regression models are tested and evaluated based on two metrics: R² (coefficient of determination) and mean absolute error (MAE).
   - The tested models are:
     - Ridge Regression
     - Elastic Net Regression
     - KNN Regression
     - Decision Tree Regression
   - The models' performance is assessed, and the best-performing models are selected based on the R² and MAE metrics.

5. Feature Importance Analysis:
   - The SHAP (SHapley Additive exPlanations) library is utilized to examine the feature importance for each selected model.
   - SHAP values provide insights into the contribution of each feature to the prediction of energy consumption and CO2 emission.
   - The feature importance analysis helps identify the key factors influencing energy consumption and CO2 emission in residential buildings.

6. Evaluation of ENERGYSTARScore:
   - The contribution of the variable "ENERGYSTARScore" to predicting CO2 emission is evaluated separately.
   - The impact of including or excluding the "ENERGYSTARScore" variable in the regression model is assessed using R² metrics.
   - This evaluation helps determine the significance of the "ENERGYSTARScore" in predicting CO2 emission and energy consumption.
