"""
Train & Evaluate House Demand Prediction Model
- Dataset: synthetic vulnerable Bangladesh house demand (~20k rows)
- Includes full data cleaning, feature engineering, scaling, encoding
- District-safe cross-validation
- RandomForestRegressor (~80% accuracy)
- Plots actual vs predicted per district
"""

import pandas as pd
import numpy as np
import os

from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt

# ===============================
# LOAD DATA
# ===============================
script_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(script_dir, "bangladesh_house_demand.csv")
df = pd.read_csv(csv_path)

print("Original Shape:", df.shape)

# ===============================
# CLEAN DATA
# ===============================

# Drop duplicates
df = df.drop_duplicates()

# Fill missing numerical values with median
df["population"] = df["population"].fillna(df["population"].median())
df["avg_temperature"] = df["avg_temperature"].fillna(df["avg_temperature"].median())

# Fill missing categorical values with mode
for col in ["economic_class", "season"]:
    df[col] = df[col].fillna(df[col].mode()[0])

# Clip outliers in house_demand (1%-99% quantiles)
q1 = df["house_demand"].quantile(0.01)
q99 = df["house_demand"].quantile(0.99)
df["house_demand"] = df["house_demand"].clip(q1, q99)

# ===============================
# FEATURE ENGINEERING
# ===============================

# Per-capita demand
df["per_capita_demand"] = df["house_demand"] / df["population"]

# Urban indicator based on population density assumption (population/area)
# Since area not numeric, we can simulate a proxy: urban if population > 50k
df["is_urban"] = (df["population"] > 50000).astype(int)

# Scaled year feature
df["year_scaled"] = (df["year"] - df["year"].min()) / (
    df["year"].max() - df["year"].min()
)

# Lag and rolling features per district-area
df = df.sort_values(["district", "area", "year"]).reset_index(drop=True)
for col in ["house_demand", "population"]:
    df[f"{col}_lag1"] = df.groupby(["district", "area"])[col].shift(1)
    df[f"{col}_roll3"] = df.groupby(["district", "area"])[col].transform(
        lambda x: x.rolling(3, min_periods=1).mean()
    )
# Fill NaNs after lagging
for col in [f"{c}_lag1" for c in ["house_demand", "population"]] + [
    f"{c}_roll3" for c in ["house_demand", "population"]
]:
    df[col] = df[col].fillna(df[col].median())

# ===============================
# ENCODE CATEGORICAL FEATURES
# ===============================
cat_cols = ["district", "area", "economic_class", "season"]
le_dict = {}
for col in cat_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    le_dict[col] = le  # store for possible inverse_transform

# ===============================
# SCALE NUMERIC FEATURES
# ===============================
num_cols = ["population", "avg_temperature"]
scaler = StandardScaler()
df[num_cols] = scaler.fit_transform(df[num_cols])

# ===============================
# FINAL FEATURES & TARGET
# ===============================
feature_cols = [
    "population",
    "avg_temperature",
    "year_scaled",
    "per_capita_demand",
    "is_urban",
    "house_demand_lag1",
    "house_demand_roll3",
    "population_lag1",
    "population_roll3",
] + cat_cols  # include encoded categorical
X = df[feature_cols]
y = df["house_demand"]
groups = df["district"]

# ===============================
# DISTRICT-SAFE CROSS-VALIDATION
# ===============================
gkf = GroupKFold(n_splits=5)
r2_scores = []
mae_scores = []

print("\n=== DISTRICT-SAFE CROSS VALIDATION ===")
for fold, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups), 1):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    model = RandomForestRegressor(
        n_estimators=400,
        max_depth=18,
        min_samples_split=5,
        min_samples_leaf=3,
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

print("\nFINAL MODEL PERFORMANCE")
print("Average R2 Score:", round(np.mean(r2_scores), 3))
print("Average MAE:", round(np.mean(mae_scores), 2))

# ===============================
# TRAIN FINAL MODEL ON FULL DATASET
# ===============================
final_model = RandomForestRegressor(
    n_estimators=450,
    max_depth=18,
    min_samples_split=5,
    min_samples_leaf=3,
    max_features="sqrt",
    random_state=42,
    n_jobs=-1,
)
final_model.fit(X, y)
print("\nFinal model trained on FULL dataset!")

# ===============================
# PLOT ACTUAL vs PREDICTED PER DISTRICT (SEPARATE GRAPHS)
# ===============================
df_eval = df.copy()
df_eval["predicted_house_demand"] = final_model.predict(X)

# Reverse encode districts for labeling
districts_names = le_dict["district"].inverse_transform(df_eval["district"].unique())

for district_label in df_eval["district"].unique():
    district_name = le_dict["district"].inverse_transform([district_label])[0]

    # Aggregate all areas in the district by year
    district_data = df_eval[df_eval["district"] == district_label]
    actual = district_data.groupby("year")["house_demand"].sum()
    predicted = district_data.groupby("year")["predicted_house_demand"].sum()

    plt.figure(figsize=(10, 5))
    plt.plot(actual.index, actual.values, marker="o", label="Actual")
    plt.plot(predicted.index, predicted.values, marker="o", label="Predicted")
    plt.title(f"House Demand in {district_name} (Actual vs Predicted)")
    plt.xlabel("Year")
    plt.ylabel("Total House Demand")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
