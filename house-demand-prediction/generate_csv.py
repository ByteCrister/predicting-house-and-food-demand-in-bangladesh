"""
Synthetic Vulnerable Bangladesh House Demand Dataset Generator (~20k rows)
- Includes districts with realistic areas, population, temperature, economic class, season
- Generates "house_demand" based on population and other features
- Introduces missing values, duplicates, and outliers
- Can be saved as CSV
"""

import pandas as pd
import numpy as np
import os

# ===============================
# SETTINGS
# ===============================
np.random.seed(42)

district_areas = {
    "Dhaka": ["Dhanmondi", "Gulshan", "Uttara", "Mirpur", "Banani"],
    "Chattogram": ["Pahartali", "Kotwali", "Double Mooring", "Chawkbazar", "Agrabad"],
    "Khulna": ["Sonadanga", "Khalishpur", "Daulatpur", "Labanchara", "Teligati"],
    "Rajshahi": ["Boalia", "Shah Makhdum", "Charghat", "Matihar", "Motihar"],
    "Barishal": ["Gournadi", "Barishal Sadar", "Babuganj", "Bakerganj", "Muladi"],
    "Sylhet": ["Zindabazar", "Beanibazar", "Balaganj", "Fenchuganj", "Kanaighat"],
    "Rangpur": ["Rangpur Sadar", "Gangachara", "Pirganj", "Badarganj", "Kaunia"],
    "Mymensingh": ["Mymensingh Sadar", "Muktagachha", "Bhaluka", "Trishal", "Phulpur"]
}

years = list(range(2010, 2026))
seasons = ["Winter", "Summer", "Monsoon", "Autumn"]
economic_classes = ["Low", "Middle", "High"]

target_rows = 20000

# ===============================
# CREATE EMPTY LIST FOR DATA
# ===============================
data = []

districts = list(district_areas.keys())

while len(data) < target_rows:
    district = np.random.choice(districts)
    area_name = np.random.choice(district_areas[district])

    base_pop = np.random.randint(5000, 100000)
    pop_growth_rate = np.random.uniform(0.01, 0.03)

    year = np.random.choice(years)
    population = int(base_pop * ((1 + pop_growth_rate) ** (year - years[0])))
    avg_temperature = round(np.random.uniform(15, 35), 1)
    economic_class = np.random.choice(economic_classes)
    season = np.random.choice(seasons)
    house_demand = int(population * np.random.uniform(0.1, 0.3))

    data.append({
        "district": district,
        "area": area_name,
        "year": year,
        "population": population,
        "avg_temperature": avg_temperature,
        "economic_class": economic_class,
        "season": season,
        "house_demand": house_demand
    })

# ===============================
# CREATE DATAFRAME
# ===============================
df = pd.DataFrame(data)

# ===============================
# INTRODUCE VULNERABILITIES
# ===============================

# 1. Missing values (~5%)
for col in ["population", "avg_temperature", "economic_class", "season"]:
    df.loc[df.sample(frac=0.05, random_state=42).index, col] = np.nan

# 2. Duplicate rows (~2%)
df = pd.concat([df, df.sample(frac=0.02, random_state=42)], ignore_index=True)

# 3. Outliers in house_demand (~1%)
outlier_idx = df.sample(frac=0.01, random_state=42).index
df.loc[outlier_idx, "house_demand"] = df.loc[outlier_idx, "house_demand"] * np.random.randint(3, 6)

# Shuffle
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# ===============================
# SAVE TO CSV
# ===============================
script_dir = os.path.dirname(os.path.abspath(__file__)) 
csv_path = os.path.join(script_dir, "bangladesh_house_demand.csv")
df.to_csv(csv_path, index=False)

print(f"Synthetic vulnerable dataset saved to {csv_path}")
print(df.head(10))
print(f"Total rows generated: {len(df)}")