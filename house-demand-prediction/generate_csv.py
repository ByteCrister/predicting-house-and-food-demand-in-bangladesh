# generate_csv.py
import os
import random
import numpy as np
import pandas as pd

# Ensure reproducible output
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

districts_areas = {
    "Dhaka": ["Dhanmondi", "Gulshan", "Uttara"],
    "Chattogram": ["Pahartali", "Double Mooring", "Kotwali"],
    "Rajshahi": ["Boalia", "Shah Makhdum", "Charghat"],
    "Khulna": ["Sonadanga", "Khalishpur", "Daulatpur"],
    "Barishal": ["Kawran Bazar", "Chandpur", "Bakerganj"],
    "Sylhet": ["Zindabazar", "Subidbazar", "Ambarkhana"],
    "Rangpur": ["Gangachara", "Mithapukur", "Kaunia"],
    "Mymensingh": ["Trishal", "Bhaluka", "Muktagachha"]
}

years = list(range(2010, 2026))  # inclusive
TARGET_ROWS = 1000

# Build at least one row per (district, area) to guarantee coverage
rows = []
all_pairs = [(d, a) for d, areas in districts_areas.items() for a in areas]
for (district, area) in all_pairs:
    year = random.choice(years)
    pop_density = int(np.random.randint(10000, 30000))
    urban_index = round(np.random.uniform(0.55, 1.0), 2)
    infra_score = round(np.random.uniform(2.0, 5.0), 2)
    econ_index = round(np.random.uniform(0.8, 1.2), 2)
    actual_demand = round(np.random.uniform(50, 250), 2)
    predicted_baseline = round(actual_demand * np.random.uniform(0.95, 1.05), 2)
    vulnerable = int(np.random.choice([0, 1, 2], p=[0.5, 0.3, 0.2]))
    rows.append([year, district, area, pop_density, urban_index, infra_score,
                 econ_index, actual_demand, predicted_baseline, vulnerable])

# Fill remaining rows up to TARGET_ROWS
while len(rows) < TARGET_ROWS:
    district, area = random.choice(all_pairs)
    year = random.choice(years)
    # Add a mild year-trend so later years have slightly higher demand on average
    year_factor = 1.0 + 0.015 * (year - 2010)
    pop_density = int(np.random.randint(10000, 30000))
    urban_index = round(np.random.uniform(0.55, 1.0), 2)
    infra_score = round(np.random.uniform(2.0, 5.0), 2)
    econ_index = round(np.random.uniform(0.8, 1.2), 2)
    base = np.random.uniform(40, 220)
    noise = np.random.normal(0, 25)
    actual_demand = round(max(0.0, base * year_factor + noise), 2)
    predicted_baseline = round(actual_demand * np.random.uniform(0.95, 1.05), 2)
    vulnerable = int(np.random.choice([0, 1, 2], p=[0.5, 0.3, 0.2]))
    rows.append([year, district, area, pop_density, urban_index, infra_score,
                 econ_index, actual_demand, predicted_baseline, vulnerable])

# Build DataFrame and shuffle for realism
df = pd.DataFrame(rows, columns=[
    "Year", "District", "Area", "Population_density", "Urbanization_index",
    "Infrastructure_score", "Economic_index", "Actual_demand", "Predicted_baseline", "Vulnerable"
])
df = df.sample(frac=1.0, random_state=RANDOM_SEED).reset_index(drop=True)

# Save CSV
out_dir = "house-demand-prediction.1.3"
os.makedirs(out_dir, exist_ok=True)
csv_path = os.path.join(out_dir, "data.csv")
df.to_csv(csv_path, index=False)

# Basic confirmation print
print(f"CSV saved at: {csv_path}")
print(f"Total rows: {len(df)}")
print("Rows per district:")
print(df.groupby("District").size().sort_values(ascending=False))
