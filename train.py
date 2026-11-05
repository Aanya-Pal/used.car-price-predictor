"""Train and compare used-car price prediction models.

Usage:  python train.py [path/to/cars.csv]
Expected columns: brand, model, year, km_driven, fuel, transmission, owner, city,
                  engine_cc, power_bhp, mileage_kmpl, seats, price_lakh
Outputs: models/best_model.joblib, outputs/metrics.csv, outputs/*.png, outputs/summary.json
"""
import json
import sys
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, RandomizedSearchCV, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBRegressor

SEED = 42
CURRENT_YEAR = 2026
TARGET = "price_lakh"
DATA = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/cars.csv")
Path("models").mkdir(exist_ok=True)
Path("outputs").mkdir(exist_ok=True)

NUM = ["age", "km_driven", "km_per_year", "engine_cc", "power_bhp", "mileage_kmpl", "seats", "owner"]
CAT = ["brand", "model", "fuel", "transmission", "city"]


# ---------------------------------------------------------------- 1. load + clean
df = pd.read_csv(DATA)
n_raw = len(df)
df = df.drop_duplicates().dropna(subset=[TARGET, "year", "km_driven"])
df = df[(df[TARGET] > 0.3) & (df["km_driven"] > 0) & (df["year"] <= CURRENT_YEAR)]
df["age"] = (CURRENT_YEAR - df["year"]).clip(lower=1)
df["km_per_year"] = df["km_driven"] / df["age"]
df = df[df["km_per_year"] < 60000]            # drops impossible odometer values
df = df[df[TARGET] < df[TARGET].quantile(0.999) * 1.5]
print(f"Rows: {n_raw} raw -> {len(df)} after cleaning")

X = df[NUM + CAT]
y_log = np.log1p(df[TARGET])                  # price is right-skewed -> model log price

# ---------------------------------------------------------------- 2. split
X_tr, X_te, ylog_tr, ylog_te = train_test_split(X, y_log, test_size=0.2, random_state=SEED)
y_te = np.expm1(ylog_te)

# ---------------------------------------------------------------- 3. preprocessing
def preprocessor(scale: bool):
    num_steps = [("imp", SimpleImputer(strategy="median"))]
    if scale:
        num_steps.append(("sc", StandardScaler()))
    return ColumnTransformer([
        ("num", Pipeline(num_steps), NUM),
        ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                          ("ohe", OneHotEncoder(handle_unknown="ignore", min_frequency=10))]), CAT),
    ])

models = {
    "Baseline (median)": Pipeline([("pre", preprocessor(False)), ("reg", DummyRegressor(strategy="median"))]),
    "Ridge": Pipeline([("pre", preprocessor(True)), ("reg", Ridge(alpha=1.0))]),
    "Random Forest": Pipeline([("pre", preprocessor(False)),
                               ("reg", RandomForestRegressor(n_estimators=300, n_jobs=-1, random_state=SEED))]),
    "XGBoost": Pipeline([("pre", preprocessor(False)),
                         ("reg", XGBRegressor(n_estimators=600, learning_rate=0.05, max_depth=5,
                                              subsample=0.8, colsample_bytree=0.8, random_state=SEED))]),
    "LightGBM": Pipeline([("pre", preprocessor(False)),
                          ("reg", LGBMRegressor(n_estimators=600, learning_rate=0.05, num_leaves=31,
                                                subsample=0.8, colsample_bytree=0.8,
                                                random_state=SEED, verbose=-1))]),
}


def score(name, pipe):
    pred = np.expm1(pipe.predict(X_te))
    return {
        "model": name,
        "MAE (lakh)": mean_absolute_error(y_te, pred),
        "RMSE (lakh)": np.sqrt(mean_squared_error(y_te, pred)),
        "R2": r2_score(y_te, pred),
        "MAPE (%)": float(np.mean(np.abs(y_te - pred) / y_te) * 100),
    }


# ---------------------------------------------------------------- 4. compare with 5-fold CV + test
kf = KFold(5, shuffle=True, random_state=SEED)
results = []
for name, pipe in models.items():
    cv = -cross_val_score(pipe, X_tr, ylog_tr, cv=kf, scoring="neg_root_mean_squared_error", n_jobs=1)
    pipe.fit(X_tr, ylog_tr)
    row = score(name, pipe)
    row["CV RMSE (log price)"] = cv.mean()
    results.append(row)
    print(f"{name:18s} CV-RMSE(log)={cv.mean():.4f}  test MAE={row['MAE (lakh)']:.3f}  R2={row['R2']:.4f}")

# ---------------------------------------------------------------- 5. tune the strongest tree model
best_before = min((r for r in results if r["model"] != "Baseline (median)"), key=lambda r: r["CV RMSE (log price)"])
print("Best before tuning:", best_before["model"])
search = RandomizedSearchCV(
    Pipeline([("pre", preprocessor(False)),
              ("reg", LGBMRegressor(random_state=SEED, verbose=-1, subsample_freq=1))]),
    {"reg__n_estimators": [300, 600, 900],
     "reg__learning_rate": [0.02, 0.05, 0.08],
     "reg__num_leaves": [15, 31, 63],
     "reg__min_child_samples": [10, 20, 40],
     "reg__subsample": [0.7, 0.85, 1.0],
     "reg__colsample_bytree": [0.7, 0.85, 1.0]},
    n_iter=12, cv=3, scoring="neg_root_mean_squared_error", random_state=SEED, n_jobs=1)
search.fit(X_tr, ylog_tr)
tuned = search.best_estimator_
row = score("LightGBM (tuned)", tuned)
row["CV RMSE (log price)"] = -search.best_score_
results.append(row)
models["LightGBM (tuned)"] = tuned
print("Tuned params:", search.best_params_)

metrics = pd.DataFrame(results).sort_values("MAE (lakh)")
metrics.to_csv("outputs/metrics.csv", index=False)
print("\n", metrics.round(4).to_string(index=False))

# ---------------------------------------------------------------- 6. pick final model on test MAE
best_name = metrics.iloc[0]["model"]
best = models[best_name]
joblib.dump(best, "models/best_model.joblib")
pred = np.expm1(best.predict(X_te))
print(f"\nFinal model: {best_name}  (saved to models/best_model.joblib)")

# ---------------------------------------------------------------- 7. plots
plt.rcParams.update({"figure.dpi": 130, "axes.spines.top": False, "axes.spines.right": False})

fig, ax = plt.subplots(figsize=(5.5, 5))
ax.scatter(y_te, pred, s=6, alpha=0.4, color="#2a6fdb")
lim = [0, max(y_te.max(), pred.max()) * 1.02]
ax.plot(lim, lim, "--", color="#555", lw=1)
ax.set(xlabel="Actual price (lakh INR)", ylabel="Predicted price (lakh INR)",
       title=f"Predicted vs actual - {best_name}")
fig.tight_layout(); fig.savefig("outputs/pred_vs_actual.png"); plt.close(fig)

fig, ax = plt.subplots(figsize=(6, 4))
resid_pct = (pred - y_te) / y_te * 100
ax.hist(resid_pct.clip(-60, 60), bins=50, color="#2a6fdb", alpha=0.85)
ax.axvline(0, color="#555", lw=1, ls="--")
ax.set(xlabel="Prediction error (%)", ylabel="Cars", title="Error distribution on test set")
fig.tight_layout(); fig.savefig("outputs/error_distribution.png"); plt.close(fig)

fig, ax = plt.subplots(figsize=(6, 3.8))
m = metrics.sort_values("MAE (lakh)", ascending=False)
ax.barh(m["model"], m["MAE (lakh)"], color="#2a6fdb")
ax.set(xlabel="Test MAE (lakh INR), lower is better", title="Model comparison")
fig.tight_layout(); fig.savefig("outputs/model_comparison.png"); plt.close(fig)

fig, ax = plt.subplots(figsize=(6, 4))
ax.hist(df[TARGET], bins=60, color="#7a8aa0")
ax.set(xlabel="Price (lakh INR)", ylabel="Cars", title="Price distribution (right-skewed)")
fig.tight_layout(); fig.savefig("outputs/price_distribution.png"); plt.close(fig)

# feature importance (permutation-free: use model's native importances grouped by original column)
reg = best.named_steps["reg"]
pre = best.named_steps["pre"]
if hasattr(reg, "feature_importances_"):
    names = pre.get_feature_names_out()
    imp = pd.Series(reg.feature_importances_, index=names)
    grouped = {}
    for n, v in imp.items():
        base = n.split("__", 1)[1]
        for c in CAT:
            if base.startswith(c + "_"):
                base = c
                break
        grouped[base] = grouped.get(base, 0) + v
    g = pd.Series(grouped).sort_values()
    g = g / g.sum()
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.barh(g.index, g.values, color="#2a6fdb")
    ax.set(xlabel="Relative importance", title="Feature importance")
    fig.tight_layout(); fig.savefig("outputs/feature_importance.png"); plt.close(fig)

# error by price band
bands = pd.cut(y_te, [0, 5, 10, 20, 50, 1000], labels=["<5L", "5-10L", "10-20L", "20-50L", ">50L"])
by_band = pd.DataFrame({"band": bands, "ape": np.abs(pred - y_te) / y_te * 100}).groupby("band", observed=True)["ape"].agg(["mean", "count"])
by_band.round(2).to_csv("outputs/error_by_price_band.csv")
print("\nMAPE by price band:\n", by_band.round(2))

summary = {
    "data_file": str(DATA), "rows_raw": n_raw, "rows_clean": int(len(df)),
    "train_rows": int(len(X_tr)), "test_rows": int(len(X_te)),
    "best_model": best_name, "best_params": {k: (v.item() if hasattr(v, "item") else v) for k, v in search.best_params_.items()},
    "test_metrics": metrics.iloc[0].drop("model").astype(float).round(4).to_dict(),
}
Path("outputs/summary.json").write_text(json.dumps(summary, indent=2))
print("\nDone.")
