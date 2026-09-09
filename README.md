# Car Price Prediction with Machine Learning 

> **Author / Created By:** Syed Shabih Ahmed (EXPS Nexus Data Science Intern)

## Overview

This repository contains the complete implementation for **Task 3: Car Price Prediction with Machine Learning** as part of the EXPS Nexus Data Science Internship. The project builds an end-to-end regression machine learning pipeline to estimate and forecast vehicle market prices based on historical specifications, brand value, and performance metrics.

## Project Workflow

* **Data Preprocessing & Cleaning:** Managed missing data rows, filtered out anomalies, and performed rigorous text standardization across vehicle attributes.
* **Feature Engineering:** Extracted predictive features such as vehicle age from manufacturing year, mileage per year ratios, and encoded categorical features (brand, fuel type, transmission) using label encoding and one-hot encoding.
* **Model Training:** Implemented multiple supervised regression models using Scikit-learn, including Linear Regression, Random Forest Regressor, and Gradient Boosting algorithms.
* **Model Evaluation:** Assessed prediction accuracy using standard statistical regression metrics including Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), and the $R^2$ Coefficient of Determination.

## Tech Stack & Prerequisites

* **Python 3.x**
* **Pandas & NumPy** (Data manipulation and numerical analysis)
* **Scikit-learn** (Model building, pipeline management, and metrics evaluation)
* **Matplotlib & Seaborn** (Visualizing feature correlations, feature importance, and actual vs. predicted price curves)

Install the required dependencies via terminal:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn

```

## Repository Structure

* `car_price_prediction.py`: Main Python script handling ingestion, data cleaning, feature engineering, model training, and evaluation.
* `car_data.csv`: Input dataset containing vehicle attributes and historical prices.
* `outputs/`: Directory containing generated evaluation graphs, correlation heatmaps, and performance visual summaries.

## Execution

Run the end-to-end pipeline script from your project root:

```bash
python car_price_prediction.py

```

## Results & Insights

* **Key Drivers:** Vehicle age, engine horsepower, and brand reputation emerged as the highest-weight features influencing market valuation.
* **Model Performance:** Ensemble methods (Random Forest and Gradient Boosting) outperformed baseline linear regression models by significantly reducing prediction error variance on unseen test splits.
