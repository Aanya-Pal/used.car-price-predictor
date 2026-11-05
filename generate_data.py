"""Generate a SYNTHETIC used-car dataset (Indian market style).

This data is artificial. It exists only so the pipeline can be built and tested
before a real dataset (e.g. CarDekho / Cars24 from Kaggle) is available.
To use real data, skip this script and put your CSV at data/cars.csv with the
same column names (see README).
"""
import numpy as np
import pandas as pd
from pathlib import Path

rng = np.random.default_rng(42)
N = 6000
CURRENT_YEAR = 2026

# brand -> (models with base ex-showroom price in INR lakh, brand value retention factor)
CATALOG = {
    "Maruti":   ({"Swift": 7.0, "Baleno": 8.0, "WagonR": 6.0, "Brezza": 11.0, "Ertiga": 11.5}, 1.00),
    "Hyundai":  ({"i20": 8.5, "Creta": 15.0, "Venue": 10.0, "Verna": 12.0, "Grand i10": 6.5}, 0.97),
    "Tata":     ({"Nexon": 10.5, "Harrier": 18.0, "Altroz": 8.5, "Punch": 8.0, "Safari": 20.0}, 0.92),
    "Honda":    ({"City": 13.0, "Amaze": 8.0, "Jazz": 9.0, "WR-V": 9.5}, 0.98),
    "Toyota":   ({"Innova Crysta": 21.0, "Fortuner": 38.0, "Glanza": 8.0, "Urban Cruiser": 11.0}, 1.05),
    "Mahindra": ({"XUV700": 20.0, "Scorpio": 15.0, "Thar": 15.5, "XUV300": 10.0, "Bolero": 10.0}, 0.99),
    "Kia":      ({"Seltos": 15.0, "Sonet": 10.5, "Carens": 12.0}, 0.96),
    "Ford":     ({"EcoSport": 10.0, "Endeavour": 33.0, "Figo": 6.0}, 0.85),
    "BMW":      ({"3 Series": 58.0, "X1": 50.0, "5 Series": 72.0}, 1.00),
    "Mercedes": ({"C-Class": 62.0, "E-Class": 82.0, "GLA": 55.0}, 1.00),
    "Audi":     ({"A4": 50.0, "Q3": 50.0, "A6": 68.0}, 1.00),
}
CITIES = {"Mumbai": 1.03, "Delhi": 1.01, "Bangalore": 1.04, "Hyderabad": 1.00, "Chennai": 0.99,
          "Pune": 1.00, "Ahmedabad": 0.97, "Kolkata": 0.95, "Jaipur": 0.96, "Lucknow": 0.94}

rows = []
brands = list(CATALOG)
brand_p = np.array([0.22, 0.17, 0.11, 0.08, 0.08, 0.08, 0.05, 0.04, 0.06, 0.06, 0.05])
brand_p = brand_p / brand_p.sum()

for _ in range(N):
    brand = rng.choice(brands, p=brand_p)
    models, retention = CATALOG[brand]
    model = rng.choice(list(models))
    base = models[model]
    age = int(np.clip(rng.gamma(2.2, 2.2), 1, 15))
    year = CURRENT_YEAR - age
    km = max(1500, rng.normal(11000, 4500) * age + rng.normal(0, 4000))
    fuel = rng.choice(["Petrol", "Diesel", "CNG"], p=[0.55, 0.40, 0.05])
    if brand in ("BMW", "Mercedes", "Audi"):
        fuel = rng.choice(["Petrol", "Diesel"], p=[0.45, 0.55])
    transmission = rng.choice(["Manual", "Automatic"], p=[0.7, 0.3] if base < 20 else [0.25, 0.75])
    owner = int(rng.choice([1, 2, 3, 4], p=[0.68, 0.24, 0.06, 0.02]))
    city = rng.choice(list(CITIES))
    engine = int(np.clip(rng.normal(1100 + base * 28, 150), 800, 3500))
    power = float(np.clip(engine * 0.075 + rng.normal(0, 6), 55, 400))
    mileage = float(np.clip(rng.normal(21 - engine / 400, 2), 8, 28))
    seats = int(rng.choice([5, 7], p=[0.8, 0.2])) if "Ertiga" in model or "Innova" in model or "Safari" in model or "XUV700" in model else 5

    # depreciation: ~20% drop in year 1, then a steady yearly decline.
    # Luxury cars depreciate faster than mass-market cars (higher running/service cost).
    luxury = brand in ("BMW", "Mercedes", "Audi")
    yearly = 0.885 if luxury else 0.915
    first_year = 0.78 if luxury else 0.82
    dep = first_year * yearly ** (age - 1)
    price = base * dep * retention * CITIES[city]
    price *= np.exp(-(km - 11000 * age) / 450000)
    price *= {1: 1.0, 2: 0.93, 3: 0.86, 4: 0.80}[owner]
    price *= 1.06 if transmission == "Automatic" else 1.0
    price *= 1.03 if fuel == "Diesel" and age < 8 else (0.95 if fuel == "Diesel" else 1.0)
    price *= 0.97 if fuel == "CNG" else 1.0
    price *= float(np.exp(rng.normal(0, 0.07)))  # market noise
    price = max(0.4, price)

    rows.append(dict(brand=brand, model=model, year=year, km_driven=int(km), fuel=fuel,
                     transmission=transmission, owner=owner, city=city, engine_cc=engine,
                     power_bhp=round(power, 1), mileage_kmpl=round(mileage, 1), seats=seats,
                     price_lakh=round(price, 2)))

df = pd.DataFrame(rows)

# Inject realistic mess so the cleaning step has something to do
idx = rng.choice(df.index, int(0.03 * N), replace=False)
df.loc[idx, "mileage_kmpl"] = np.nan
idx = rng.choice(df.index, int(0.02 * N), replace=False)
df.loc[idx, "power_bhp"] = np.nan
idx = rng.choice(df.index, int(0.01 * N), replace=False)
df.loc[idx, "km_driven"] = df.loc[idx, "km_driven"] * 10      # entry errors
dups = df.sample(40, random_state=1)
df = pd.concat([df, dups], ignore_index=True)

Path("data").mkdir(exist_ok=True)
df.to_csv("data/cars.csv", index=False)
print(f"Wrote data/cars.csv with {len(df)} rows")
print(df.describe(include="all").T[["count", "mean", "min", "max"]].to_string())
