# Used Car Price Prediction

Predicts the resale price (in lakh INR) of a used car from its age, kilometres driven,
brand/model, fuel, transmission, owners, city and engine details.

> **Data note:** the bundled `data/cars.csv` is **synthetic** (made by `generate_data.py`)
> so the pipeline could be built before a real dataset was available. Scores on it are
> optimistic and do not reflect real market prices. Replace it with real listings
> (e.g. CarDekho / Cars24 from Kaggle) and retrain.

## Pipeline

1. **Clean:** drop duplicates, impossible odometer readings, and extreme prices
2. **Features:** car age, km per year, one-hot encoded categoricals, median/mode imputation
3. **Target:** `log(1 + price)` because prices are right-skewed
4. **Models compared:** median baseline, Ridge, Random Forest, XGBoost, LightGBM, tuned LightGBM
5. **Evaluation:** 5-fold CV on the training split, final scores on a held-out 20% test set
6. **Outputs:** saved model, metrics table, plots

## Run it

```bash
pip install -r requirements.txt
python generate_data.py        # skip if you have your own data/cars.csv
python train.py                # or: python train.py path/to/your.csv
streamlit run app.py
```

## Using a real dataset

Map your CSV to these columns (rename in pandas or edit `NUM`/`CAT` in `train.py`):

`brand, model, year, km_driven, fuel, transmission, owner, city, engine_cc, power_bhp, mileage_kmpl, seats, price_lakh`

## Results (on synthetic data)

See `outputs/metrics.csv` and the plots in `outputs/`.

## Project layout

```
generate_data.py   synthetic data generator
train.py           cleaning, features, training, evaluation, plots
app.py             Streamlit price estimator
models/            saved pipeline (best_model.joblib)
outputs/           metrics, summary.json, figures
data/cars.csv      dataset
```
