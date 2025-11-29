#!/usr/bin/env python3
"""
Generate EXTREMELY VULNERABLE synthetic district-level food demand data for Bangladesh (2010-2025)
Designed for FULL feature engineering, leakage removal, cleaning, encoding, scaling, outlier handling.
"""

import pandas as pd
import numpy as np
import random
import string

np.random.seed(42)

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
economic_classes = ["low", "middle", "high", None]
years = list(range(2010, 2026))
seasons = ["summer", "monsoon", "winter", None]

data = []

for district, areas in districts_areas.items():
    for area in areas:
        base_pop = np.random.randint(100_000, 900_000)
        base_density = np.random.uniform(300, 7000)

        for year in years:
            growth_rate = np.random.uniform(0.01, 0.02)
            pop = base_pop * ((1 + growth_rate) ** (year - 2010))

            food_demand = {}
            for food in food_types:
                value = np.random.normal(60, 25)

                if np.random.rand() < 0.07:
                    value *= random.choice([0.05, 4, 10])

                if np.random.rand() < 0.06:
                    value = np.nan

                food_demand[food] = max(0, value)

            population = int(pop) if np.random.rand() > 0.03 else np.nan
            pop_density = round(base_density, 1) if np.random.rand() > 0.04 else np.nan

            useless_string = "".join(random.choices(string.ascii_letters, k=8))
            leakage_profit = np.random.randint(5000, 80000)  # DATA LEAKAGE COLUMN

            mixed_temp = (
                np.random.uniform(25, 40)
                if np.random.rand() > 0.5
                else np.random.uniform(70, 110)
            )

            data.append(
                {
                    "district": (
                        district if np.random.rand() > 0.01 else district.lower()
                    ),
                    "area": area if np.random.rand() > 0.01 else None,
                    "year": year if np.random.rand() > 0.02 else str(year),
                    "population": population,
                    "pop_density": pop_density,
                    "economic_class": random.choice(economic_classes),
                    "season": random.choice(seasons),
                    "avg_temperature": mixed_temp,
                    "random_text": useless_string,
                    "duplicate_noise": random.choice([1, 1, 1, 999]),
                    "leakage_future_profit": leakage_profit,
                    **food_demand,
                }
            )

df = pd.DataFrame(data)

df = pd.concat([df, df.sample(80)], ignore_index=True)

script_dir = os.path.dirname(os.path.abspath(__file__)) 
csv_path = os.path.join(script_dir, "bangladesh_food_demand.csv")
df.to_csv(csv_path, index=False)

print("✅ EXTREMELY vulnerable dataset created successfully!")
print(csv_path)
