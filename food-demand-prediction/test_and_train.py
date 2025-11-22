#!/usr/bin/env python3
"""
Train & evaluate a Regularized RandomForest model to predict district-level food demand
with feature engineering, per-district cross-validation, and evaluation metrics.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import GroupKFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# --- Load CSV ---
csv_path = r"food-demand-prediction/bangladesh_food_demand.csv"
df = pd.read_csv(csv_path)

# --- Feature Engineering & Data Cleaning ---
num_cols = ["population", "pop_density", "rice", "wheat", "pulses", "meat", "fish", "vegetables"]

# 1. Handle missing values for numeric columns
df[num_cols] = df[num_cols].fillna(df[num_cols].median())

# 2. Outlier capping (IQR method)
for col in num_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    df[col] = np.clip(df[col], lower, upper)

# 3. Feature engineering
df["pop_density_per_1000"] = df["pop_density"] / 1000
df["year_scaled"] = (df["year"] - df["year"].min()) / (df["year"].max() - df["year"].min())

# 4. Lag features: previous year food demand per district and area
df_sorted = df.sort_values(["district", "area", "year"])
for food in ["rice", "wheat", "pulses", "meat", "fish", "vegetables"]:
    df_sorted[f"{food}_prev"] = df_sorted.groupby(["district", "area"])[food].shift(1)

# 5. Fill NaNs in numeric columns only (including lag features)
numeric_cols = num_cols + [f"{food}_prev" for food in ["rice", "wheat", "pulses", "meat", "fish", "vegetables"]]
df_sorted[numeric_cols] = df_sorted[numeric_cols].fillna(df_sorted[numeric_cols].median())

# --- Features & Targets ---
features = ["population", "pop_density", "pop_density_per_1000", "year_scaled",
            "rice_prev", "wheat_prev", "pulses_prev", "meat_prev", "fish_prev", "vegetables_prev"]
targets = ["rice", "wheat", "pulses", "meat", "fish", "vegetables"]

X = df_sorted[features]
y = df_sorted[targets]
groups = df_sorted["district"]

# --- Grouped cross-validation per district ---
gkf = GroupKFold(n_splits=len(df["district"].unique()))
print("=== Per-District Cross-Validation Metrics ===")

for train_idx, test_idx in gkf.split(X, y, groups=groups):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    district_name = groups.iloc[test_idx].unique()[0]

    # --- Train Regularized Random Forest ---
    model = RandomForestRegressor(
        n_estimators=300,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=3,
        max_features='sqrt',
        random_state=42
    )
    model.fit(X_train, y_train)

    # --- Predict & Evaluate ---
    y_pred = model.predict(X_test)
    print(f"\nDistrict: {district_name}")
    for i, food in enumerate(targets):
        mse = mean_squared_error(y_test[food], y_pred[:, i])
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test[food], y_pred[:, i])
        r2 = r2_score(y_test[food], y_pred[:, i])
        print(f"{food}: MSE={mse:.2f}, RMSE={rmse:.2f}, MAE={mae:.2f}, R2={r2:.2f}")

# --- Train final model on full dataset ---
final_model = RandomForestRegressor(
    n_estimators=300,
    max_depth=10,
    min_samples_split=5,
    min_samples_leaf=3,
    max_features='sqrt',
    random_state=42
)
final_model.fit(X, y)

# --- Predict on full dataset ---
df_pred = df_sorted.copy()
df_pred[targets] = final_model.predict(X)

# --- Aggregate actual and predicted by district ---
df_actual = df_sorted.groupby(["district", "year"])[targets].sum().reset_index()
df_district = df_pred.groupby(["district", "year"])[targets].sum().reset_index()

# --- Plot per district per food with metrics ---
for food in targets:
    for district in df_actual["district"].unique():
        subset_actual = df_actual[df_actual["district"] == district]
        subset_pred = df_district[df_district["district"] == district]

        # --- Metrics per district per food ---
        mse = mean_squared_error(subset_actual[food], subset_pred[food])
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(subset_actual[food], subset_pred[food])
        r2 = r2_score(subset_actual[food], subset_pred[food])
        print(f"\nFinal Metrics - District: {district}, Food: {food}")
        print(f"MSE={mse:.2f}, RMSE={rmse:.2f}, MAE={mae:.2f}, R2={r2:.2f}")

        # --- Plot ---
        plt.figure(figsize=(8, 5))
        plt.plot(subset_actual["year"], subset_actual[food], 'o--', color='blue', label='Actual')
        plt.plot(subset_pred["year"], subset_pred[food], 'o-', color='orange', label='Predicted')
        plt.title(f"{food.capitalize()} Demand: {district} (2010-2025)")
        plt.xlabel("Year")
        plt.ylabel(f"{food.capitalize()} Demand (kg/year)")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()
