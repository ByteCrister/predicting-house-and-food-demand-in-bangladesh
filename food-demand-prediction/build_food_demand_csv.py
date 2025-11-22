#!/usr/bin/env python3
"""
Generate synthetic district-level population & food demand data for Bangladesh (2010-2025)
with district and area columns, including missing values and anomalies for cleaning/feature engineering practice.
"""

import pandas as pd
import numpy as np

# --- Districts & corresponding areas ---
districts_areas = {
    "Dhaka": ["Dhaka North", "Dhaka South", "Gazipur", "Narsingdi"],
    "Chittagong": ["Chittagong City", "Cox's Bazar", "Feni", "Comilla"],
    "Khulna": ["Khulna City", "Jessore", "Satkhira", "Bagerhat"],
    "Rajshahi": ["Rajshahi City", "Pabna", "Natore", "Bogra"],
    "Barisal": ["Barisal City", "Patuakhali", "Bhola", "Jhalokathi"],
    "Sylhet": ["Sylhet City", "Moulvibazar", "Habiganj", "Sunamganj"],
    "Rangpur": ["Rangpur City", "Dinajpur", "Thakurgaon", "Lalmonirhat"],
    "Mymensingh": ["Mymensingh City", "Netrokona", "Jamalpur", "Sherpur"],
}

food_types = ["rice", "wheat", "pulses", "meat", "fish", "vegetables"]
years = list(range(2010, 2026))

# --- Generate synthetic data ---
data = []
np.random.seed(42)

for district, areas in districts_areas.items():
    for area in areas:
        base_pop = np.random.randint(100_000, 1_000_000)
        base_density = np.random.uniform(500, 5000)
        for year in years:
            growth_rate = np.random.uniform(0.011, 0.015)
            pop = base_pop * ((1 + growth_rate) ** (year - 2010))

            # Food demand per person (kg/year) with random variation
            food_demand = {}
            for food in food_types:
                value = np.random.normal(
                    loc=base_density / 1000 + np.random.uniform(10, 50), scale=5
                )
                
                # Introduce outliers
                if np.random.rand() < 0.05:  # 5% chance
                    value *= np.random.choice([0.1, 3, 5])  # extreme low or high
                # Introduce missing values
                if np.random.rand() < 0.03:  # 3% chance
                    value = np.nan
                    
                food_demand[food] = max(0, value)

            # Introduce occasional missing population
            population_val = int(pop)
            if np.random.rand() < 0.02:
                population_val = np.nan

            # Introduce occasional missing density
            density_val = round(base_density, 2)
            if np.random.rand() < 0.02:
                density_val = np.nan

            data.append(
                {
                    "district": district,
                    "area": area,
                    "year": year,
                    "population": population_val,
                    "pop_density": density_val,
                    **food_demand,
                }
            )

# --- Create DataFrame and save CSV ---
df = pd.DataFrame(data)
csv_path = r"C:\Users\WD-OLY\OneDrive\Desktop\test\food-demand-prediction\bangladesh_food_demand.csv"
df.to_csv(csv_path, index=False)

print(f"Vulnerable CSV file created successfully in:\n{csv_path}")
print("It includes missing values and some extreme outliers for cleaning/feature engineering practice.")
