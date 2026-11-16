# Building Energy Efficiency Prediction Using Multi-Output Regression

## Project Overview
 - This project explores machine learning techniques to predict building energy performance using the UCI Machine Learning Repository's Energy Efficiency dataset.
 - The objective is to predict two target variables, Heating Load and Cooling Load, based on building design characteristics. Multiple regression algorithms are compared to evaluate predictive performance and examine the impact of feature selection and scaling.

## Dataset:
 - Source: UCI Energy Efficiency Dataset
   - The dataset contains 768 building configurations with eight input features, including relative compactness, surface area, wall area, roof area, overall height, orientation, glazing area, and glazing area distribution.
   - Target Variables:
     - Heating Load (Y1)
     - Cooling Load (Y2)

## Methodology:
 - Performed exploratory data analysis (EDA) to understand feature distributions and relationships.
 - Explored feature selection and scaling techniques.
 - Implemented and compared multiple regression algorithms:
   - Linear Regression
   - Decision Tree Regressor
   - Random Forest Regressor
   - K-Nearest Neighbors (KNN)
   - Support Vector Regression (SVR) with MultiOutput
   - LinearSVR
 - Evaluated model performance using R², Mean Absolute Error (MAE), and Root Mean Squared Error (RMSE).
 - Developed a reusable train_and_test_models() function for model training and comparison.

## Tools & Technologies
 - Python | Pandas | NumPy | Scikit-learn | Matplotlib | Jupyter Notebook

## Key Takeaways
 - Compared linear and nonlinear regression approaches for predicting building heating and cooling loads.
 - Investigated the influence of feature selection and scaling on model performance.
 - Explored Random Forest as a prominent modeling approach for multi-output energy prediction.
 - Demonstrated the application of machine learning to building energy efficiency analysis.

## Applications
 - This project demonstrates how predictive modeling can support building energy performance assessment and provide a foundation for further research in smart buildings, energy optimization, and intelligent building management systems.

## Project Type
 - Master's Academic Project — Data Science
