"""
CAR PRICE PREDICTION
---------------------
Goal: Predict the selling price of a car using its specifications
(engine size, horsepower, mileage, brand, etc.)

Dataset: CarPrice.csv (205 cars, 26 columns)
Source : https://raw.githubusercontent.com/amankharwal/Website-data/master/CarPrice.csv

"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# ---------------------------------------------------------------
# SETUP
# ---------------------------------------------------------------
DATA_PATH = "car_prices.csv"         
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

sns.set_style("whitegrid")


# ---------------------------------------------------------------
# LOAD THE DATA AND TAKE A FIRST LOOK
# ---------------------------------------------------------------
print("STEP 1: Loading data...")
df = pd.read_csv(DATA_PATH)

print(f"Shape of the dataset: {df.shape}")
print("\nFirst 5 rows:")
print(df.head())

print("\nColumn info:")
print(df.info())

print("\nMissing values per column:")
print(df.isnull().sum())
# Note: this dataset does not have blank/NaN cells, but it DOES have
# messy text in the CarName column (typos, inconsistent brand spellings).
# We fix that in Step 2 -- that's our real "cleaning" job here.

print("\nDuplicate rows:", df.duplicated().sum())


# ---------------------------------------------------------------
# CLEAN THE DATA
# ---------------------------------------------------------------
print("\nSTEP 2: Cleaning the data...")

# Extract the brand and fix the typos manually.

df["brand"] = df["CarName"].apply(lambda x: x.split(" ")[0].lower())

brand_corrections = {
    "maxda": "mazda",
    "porcshce": "porsche",
    "toyouta": "toyota",
    "vokswagen": "volkswagen",
    "vw": "volkswagen",
    "nissan": "nissan", 
}

df["brand"] = df["brand"].replace(brand_corrections)

print("Unique brands after cleaning:")
print(sorted(df["brand"].unique()))

df = df.drop(columns=["car_ID", "CarName"])


# ---------------------------------------------------------------
# EXPLORATORY DATA ANALYSIS (EDA)
# ---------------------------------------------------------------
print("\nSTEP 3: Exploring the data...")

plt.figure(figsize=(8, 5))
sns.histplot(df["price"], kde=True, color="steelblue")
plt.title("Distribution of Car Prices")
plt.xlabel("Price ($)")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/price_distribution.png")
plt.close()

numeric_df = df.select_dtypes(include=[np.number])
plt.figure(figsize=(12, 9))
sns.heatmap(numeric_df.corr(), cmap="coolwarm", annot=False)
plt.title("Correlation Heatmap of Numeric Features")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/correlation_heatmap.png")
plt.close()

top_corr = numeric_df.corr()["price"].sort_values(ascending=False)
print("\nTop features correlated with price:")
print(top_corr.head(10))

plt.figure(figsize=(12, 6))
brand_avg_price = df.groupby("brand")["price"].mean().sort_values(ascending=False)
sns.barplot(x=brand_avg_price.values, y=brand_avg_price.index, color="steelblue")
plt.title("Average Price by Brand")
plt.xlabel("Average Price ($)")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/avg_price_by_brand.png")
plt.close()

print("Charts saved in the 'outputs' folder.")


# ---------------------------------------------------------------
# FEATURE ENGINEERING (PREP FOR MODELING)
# ---------------------------------------------------------------
print("\nSTEP 4: Preparing features for the model...")

X = df.drop(columns=["price"])
y = df["price"]

X = pd.get_dummies(X, drop_first=True)

print(f"Final feature matrix shape: {X.shape}")


# ---------------------------------------------------------------
# TRAIN / TEST SPLIT
# ---------------------------------------------------------------
print("\nSTEP 5: Splitting into train and test sets...")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"Training samples: {len(X_train)} | Testing samples: {len(X_test)}")


# ---------------------------------------------------------------
# TRAIN THE MODELS
# ---------------------------------------------------------------
print("\nSTEP 6: Training models...")

# Model 1: Linear Regression (simple, interpretable baseline)
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
lr_preds = lr_model.predict(X_test)

# Model 2: Random Forest (usually performs better on this kind of data)
rf_model = RandomForestRegressor(n_estimators=200, random_state=42)
rf_model.fit(X_train, y_train)
rf_preds = rf_model.predict(X_test)


# ---------------------------------------------------------------
# EVALUATE THE MODELS
# ---------------------------------------------------------------
def evaluate(name, y_true, y_pred):
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    print(f"\n{name} Results:")
    print(f"  R2 Score : {r2:.4f}  (closer to 1.0 is better)")
    print(f"  MAE      : {mae:.2f}  (average error in dollars)")
    print(f"  RMSE     : {rmse:.2f}")
    return r2, mae, rmse


print("\nSTEP 7: Evaluating models on the test set...")
evaluate("Linear Regression", y_test, lr_preds)
evaluate("Random Forest", y_test, rf_preds)

# Visualization
plt.figure(figsize=(7, 7))
plt.scatter(y_test, rf_preds, alpha=0.6, color="steelblue")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--")
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Random Forest: Actual vs Predicted Price")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/actual_vs_predicted_price.png")
plt.close()


# ---------------------------------------------------------------
# WHICH FEATURES MATTER MOST?
# ---------------------------------------------------------------
print("\nSTEP 8: Feature importance (Random Forest)...")

importance = pd.Series(rf_model.feature_importances_, index=X.columns)
importance = importance.sort_values(ascending=False).head(10)

plt.figure(figsize=(10, 6))
sns.barplot(x=importance.values, y=importance.index, color="steelblue")
plt.title("Top 10 Most Important Features for Predicting Price")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/feature_importance.png")
plt.close()

print(importance)
print("\nDone! Check the 'outputs' folder for all charts.")
