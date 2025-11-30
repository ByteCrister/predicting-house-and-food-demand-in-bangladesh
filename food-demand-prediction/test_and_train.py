#!/usr/bin/env python3
"""
FULL training & evaluation pipeline for EXTREMELY dirty food demand dataset
with advanced feature engineering, strong cleaning, lag features,
district-safe GroupKFold, and high-accuracy RandomForest.
"""

import pandas as pd
import numpy as np
import os

from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

import matplotlib.pyplot as plt

# ============================================================
# LOAD DATASET
# ============================================================
script_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(script_dir, "bangladesh_food_demand.csv")
df = pd.read_csv(csv_path)

print("\nOriginal Shape:", df.shape)

# ============================================================
# FIX DATA TYPES
# ============================================================
df["year"] = pd.to_numeric(df["year"], errors="coerce")
df["district"] = df["district"].astype(str).str.strip().str.title()

# ============================================================
# REMOVE LEAKAGE & TRASH COLUMNS (ONLY IF THEY EXIST)
# ============================================================
drop_cols = ["random_text", "duplicate_noise", "leakage_future_profit"]
df = df.drop(columns=[c for c in drop_cols if c in df.columns])

# ============================================================
# NUMERIC & CATEGORICAL COLUMNS
# ============================================================
food_cols = ["rice", "wheat", "pulses", "meat", "fish", "vegetables"]
num_cols = ["population", "pop_density", "avg_temperature"] + food_cols
cat_cols = ["district", "area", "economic_class", "season"]

# ============================================================
# HANDLE MISSING VALUES
# ============================================================
for col in num_cols:
    df[col] = df[col].fillna(df[col].median())

for col in cat_cols:
    df[col] = df[col].fillna(df[col].mode()[0])

# ============================================================
# EXTREME OUTLIER CLIPPING (1%–99%)
# ============================================================
for col in num_cols:
    q1 = df[col].quantile(0.01)
    q99 = df[col].quantile(0.99)
    df[col] = df[col].clip(q1, q99)

# ============================================================
# SORT FOR TIME FEATURES
# ============================================================
df = df.sort_values(["district", "area", "year"]).reset_index(drop=True)

# ============================================================
# ADVANCED FEATURE ENGINEERING
# ============================================================
df["protein_demand"] = df["meat"] + df["fish"] + df["pulses"]
df["total_food_demand"] = df[food_cols].sum(axis=1)
df["per_capita_demand"] = df["total_food_demand"] / df["population"]
df["is_urban"] = (df["pop_density"] > 2500).astype(int)

df["year_scaled"] = (df["year"] - df["year"].min()) / (
    df["year"].max() - df["year"].min()
)

# ============================================================
# SAFE LAG + ROLLING FEATURES (NO INDEX ERRORS)
# ============================================================
for food in food_cols:
    df[f"{food}_lag1"] = df.groupby(["district", "area"])[food].shift(1)

    df[f"{food}_roll3"] = df.groupby(["district", "area"])[food].transform(
        lambda x: x.rolling(3, min_periods=1).mean()
    )

lag_cols = [f"{f}_lag1" for f in food_cols] + [f"{f}_roll3" for f in food_cols]

for col in lag_cols:
    df[col] = df[col].fillna(df[col].median())

df["district_name"] = df["district"].copy()

# ============================================================
# ENCODING CATEGORICAL FEATURES
# ============================================================
le = LabelEncoder()
for col in cat_cols:
    df[col] = le.fit_transform(df[col])

# ============================================================
# SCALING NUMERICAL FEATURES
# ============================================================
scale_cols = ["population", "pop_density", "avg_temperature"]
scaler = StandardScaler()
df[scale_cols] = scaler.fit_transform(df[scale_cols])

# ============================================================
# FINAL FEATURE SET & TARGET
# ============================================================
feature_cols = (
    ["population", "pop_density", "avg_temperature", "year_scaled", "is_urban"]
    + lag_cols
    + ["protein_demand", "per_capita_demand"]
)

X = df[feature_cols]
y = df["total_food_demand"]
groups = df["district"]

# ============================================================
# DISTRICT-SAFE GROUP K-FOLD
# ============================================================
gkf = GroupKFold(n_splits=5)

r2_scores = []
mae_scores = []

print("\n=== DISTRICT-SAFE CROSS VALIDATION ===")

for fold, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups), 1):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    model = RandomForestRegressor(
        n_estimators=350,
        max_depth=14,
        min_samples_split=6,
        min_samples_leaf=5,
        max_features="sqrt",
        random_state=42,
        n_jobs=-1,
    )

    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    r2 = r2_score(y_test, preds)
    mae = mean_absolute_error(y_test, preds)

    r2_scores.append(r2)
    mae_scores.append(mae)

    print(f"Fold {fold} → R2: {r2:.3f} | MAE: {mae:.2f}")

# ============================================================
# FINAL PERFORMANCE
# ============================================================
print("\nFINAL MODEL PERFORMANCE")
print("Average R2 Score:", round(np.mean(r2_scores), 3))
print("Average MAE:", round(np.mean(mae_scores), 2))

# ============================================================
# TRAIN FINAL MODEL ON FULL DATASET
# ============================================================
final_model = RandomForestRegressor(
    n_estimators=400,
    max_depth=14,
    min_samples_split=6,
    min_samples_leaf=5,
    max_features="sqrt",
    random_state=42,
    n_jobs=-1,
)

final_model.fit(X, y)

print("\n✅ Final model trained on FULL dataset!")


# ============================================================
# FUNCTION 1: ACTUAL vs PREDICTED (DISTRICT-WISE FIGURES)
# ============================================================
def show_actual_vs_predicted_per_district_plot(df, model, feature_cols, food_cols):
    df_eval = df.copy()
    df_eval["predicted_total"] = model.predict(df_eval[feature_cols])

    food_ratio = df[food_cols].div(df[food_cols].sum(axis=1), axis=0)

    for food in food_cols:
        df_eval[f"pred_{food}"] = df_eval["predicted_total"] * food_ratio[food]

    actual = df.groupby(["district_name", "year"])[food_cols].sum().reset_index()

    predicted = (
        df_eval.groupby(["district_name", "year"])[[f"pred_{f}" for f in food_cols]]
        .sum()
        .reset_index()
    )

    districts = actual["district_name"].unique()

    for district in districts:
        actual_d = actual[actual["district_name"] == district]
        pred_d = predicted[predicted["district_name"] == district]

        for food in food_cols:
            plt.figure(figsize=(8, 5))

            plt.plot(actual_d["year"], actual_d[food])
            plt.plot(pred_d["year"], pred_d[f"pred_{food}"])

            plt.title(f"{district} - {food.capitalize()} Demand (Actual vs Predicted)")
            plt.xlabel("Year")
            plt.ylabel("Demand")
            plt.legend(["Actual", "Predicted"])
            plt.grid(True)
            plt.tight_layout()
            plt.show()


# ============================================================
# FUNCTION 2: FUTURE FOOD DEMAND (BANGLADESH FIGURE)
# ============================================================
def predict_future_bangladesh_plot(
    df, model, feature_cols, food_cols, start_year=2026, end_year=2035
):
    df_future = df.copy()

    # Extract last known state per district-area
    last_rows = (
        df_future.sort_values("year")
        .groupby(["district", "area"])
        .tail(1)
        .reset_index(drop=True)
    )

    future_all = []

    for year in range(start_year, end_year + 1):

        temp = last_rows.copy()

        # Year update + scaling
        temp["year"] = year
        temp["year_scaled"] = (year - df["year"].min()) / (
            df["year"].max() - df["year"].min()
        )

        # Population growth model (1.1% annual)
        temp["population"] = temp["population"] * 1.011

        # Temperature drift
        temp["avg_temperature"] = temp["avg_temperature"] + 0.02

        # Lag update using PREVIOUS PREDICTION, not real values
        for food in food_cols:
            temp[f"{food}_lag1"] = temp[f"{food}_roll3"]

        # Predict total food demand
        temp["predicted_total"] = model.predict(temp[feature_cols])

        # Dynamic national food ratios (trend preserving)
        rolling_ratio = (
            df.groupby("year")[food_cols].sum().tail(5).mean()
            / df.groupby("year")[food_cols].sum().tail(5).mean().sum()
        )

        for food in food_cols:
            temp[food] = temp["predicted_total"] * rolling_ratio[food]

        # Recompute engineered features (ABSOLUTELY REQUIRED)
        temp["protein_demand"] = temp["meat"] + temp["fish"] + temp["pulses"]
        temp["total_food_demand"] = temp[food_cols].sum(axis=1)
        temp["per_capita_demand"] = temp["total_food_demand"] / temp["population"]

        # Roll forward
        last_rows = temp.copy()
        future_all.append(temp)

    future_df = pd.concat(future_all, ignore_index=True)

    # NATIONAL AGGREGATION
    national_forecast = future_df.groupby("year")[food_cols].sum().reset_index()

    # NATIONAL PLOT
    plt.figure(figsize=(12, 7))

    for food in food_cols:
        plt.plot(national_forecast["year"], national_forecast[food], linewidth=2)

    plt.title("Corrected Future Food Demand Forecast for Bangladesh")
    plt.xlabel("Year")
    plt.ylabel("Total National Demand")
    plt.legend(food_cols)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    return national_forecast


show_actual_vs_predicted_per_district_plot(df, final_model, feature_cols, food_cols)
# predict_future_bangladesh_plot(df, final_model, feature_cols, food_cols)
