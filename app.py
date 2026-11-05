"""Streamlit app: estimate the resale price of a used car.

Run:  streamlit run app.py
"""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

CURRENT_YEAR = 2026
ROOT = Path(__file__).parent


def find(folder, name):
    """Return folder/name if it exists, else name next to app.py (works for flat uploads)."""
    for cand in (ROOT / folder / name, ROOT / name):
        if cand.exists():
            return cand
    return ROOT / folder / name

st.set_page_config(page_title="AutoWorth | Used Car Price Predictor", page_icon="🚗", layout="wide")

# ------------------------------------------------------------------ styling
st.markdown(
    """
    <style>
      .block-container {padding-top: 4rem; max-width: 1150px;}
      #MainMenu, footer {visibility: hidden;}
      .hero {
        background: linear-gradient(120deg, #0e7490 0%, #4f46e5 55%, #9333ea 100%);
        border-radius: 20px; padding: 2.2rem 2.4rem; margin-bottom: 1.4rem;
        box-shadow: 0 10px 40px rgba(79,70,229,.35);
      }
      .hero h1 {margin: 0; font-size: 2.5rem; color: #fff; letter-spacing: -0.5px;}
      .hero p {margin: .5rem 0 0; color: #e0e7ff; font-size: 1.05rem;}
      .chip {display:inline-block; padding:.25rem .8rem; border-radius:999px; margin:.9rem .4rem 0 0;
             background: rgba(255,255,255,.16); color:#fff; font-size:.82rem;}
      .price-card {
        background: linear-gradient(160deg, #131c31, #0f172a); border: 1px solid #243352;
        border-radius: 20px; padding: 1.8rem; text-align: center;
        box-shadow: 0 8px 30px rgba(0,0,0,.35);
      }
      .price-card .label {color:#94a3b8; font-size:.9rem; letter-spacing:.12em; text-transform:uppercase;}
      .price-card .value {font-size: 3.4rem; font-weight: 800; color:#22d3ee; line-height:1.1; margin:.3rem 0;}
      .price-card .range {color:#cbd5e1; font-size:1rem;}
      .bar {height:10px; border-radius:99px; background:#1e293b; margin:1.1rem 0 .4rem; position:relative;}
      .bar > span {position:absolute; top:0; height:10px; border-radius:99px;
                   background: linear-gradient(90deg,#22d3ee,#818cf8);}
      .bar-labels {display:flex; justify-content:space-between; color:#94a3b8; font-size:.8rem;}
      .stat {background:#131c31; border:1px solid #243352; border-radius:16px; padding:1rem 1.2rem;}
      .stat .n {font-size:1.7rem; font-weight:700; color:#e6edf7;}
      .stat .t {color:#94a3b8; font-size:.85rem;}
      div[data-testid="stTabs"] button {font-size: 1rem;}
      .stButton > button {border-radius: 12px; font-weight: 700; padding: .7rem 1rem;}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_model():
    return joblib.load(find("models", "best_model.joblib"))


@st.cache_data
def load_reference():
    return pd.read_csv(find("data", "cars.csv"))


@st.cache_data
def load_summary():
    p = find("outputs", "summary.json")
    return json.loads(p.read_text()) if p.exists() else {}


model = load_model()
ref = load_reference()
summary = load_summary()
tm = summary.get("test_metrics", {})


def build_row(age, km, engine, power, mileage, seats, owner, brand, model_name, fuel, trans, city):
    return {
        "age": age, "km_driven": km, "km_per_year": km / max(age, 1), "engine_cc": engine,
        "power_bhp": power, "mileage_kmpl": mileage, "seats": seats, "owner": owner,
        "brand": brand, "model": model_name, "fuel": fuel, "transmission": trans, "city": city,
    }


def predict(rows):
    return np.expm1(model.predict(pd.DataFrame(rows)))


# ------------------------------------------------------------------ hero
mae = tm.get("MAE (lakh)")
mape = tm.get("MAPE (%)")
st.markdown(
    f"""
    <div class="hero">
      <h1>🚗 AutoWorth</h1>
      <p>Know what your used car is really worth, in seconds.</p>
      <span class="chip">LightGBM model</span>
      <span class="chip">{len(ref):,} cars trained on</span>
      {f'<span class="chip">~{mape:.1f}% average error</span>' if mape else ''}
    </div>
    """,
    unsafe_allow_html=True,
)

tab_predict, tab_insights, tab_about = st.tabs(["💰 Price estimate", "📊 Model insights", "ℹ️ About"])

# ------------------------------------------------------------------ predict tab
with tab_predict:
    left, right = st.columns([1.05, 1], gap="large")

    with left:
        st.subheader("Car details")
        c1, c2 = st.columns(2)
        brand = c1.selectbox("Brand", sorted(ref["brand"].unique()))
        model_name = c2.selectbox("Model", sorted(ref.loc[ref["brand"] == brand, "model"].unique()))
        similar = ref[(ref["brand"] == brand) & (ref["model"] == model_name)]

        year = c1.number_input("Manufacturing year", 2008, CURRENT_YEAR, 2020)
        km = c2.number_input("Kilometres driven", 500, 400000, 40000, step=1000)
        # only offer fuel / transmission options that exist for this model, most common first
        fuel_opts = similar["fuel"].value_counts().index.tolist() or sorted(ref["fuel"].unique())
        trans_opts = similar["transmission"].value_counts().index.tolist() or sorted(ref["transmission"].unique())
        fuel = c1.selectbox("Fuel", fuel_opts)
        trans = c2.selectbox("Transmission", trans_opts)
        owner = c1.selectbox("Previous owners", [1, 2, 3, 4])
        city = c2.selectbox("City", sorted(ref["city"].unique()))

        with st.expander("Engine details (auto-filled from similar cars)"):
            e1, e2 = st.columns(2)
            engine = e1.number_input("Engine (cc)", 600, 4000, int(similar["engine_cc"].median()))
            power = e2.number_input("Power (bhp)", 40.0, 450.0, float(similar["power_bhp"].median()))
            mileage = e1.number_input("Mileage (kmpl)", 5.0, 35.0, float(similar["mileage_kmpl"].median()))
            seats = e2.selectbox("Seats", [5, 7])

        go = st.button("Estimate price", type="primary", width="stretch")

    with right:
        st.subheader("Estimate")
        age = max(1, CURRENT_YEAR - year)
        args = (km, engine, power, mileage, seats, owner, brand, model_name, fuel, trans, city)

        if go or "last" in st.session_state:
            if go:
                st.session_state["last"] = True
            price = float(predict([build_row(age, *args)])[0])
            low, high = price * 0.90, price * 1.10

            cheapest = float(similar[similar["year"] >= year - 3]["price_lakh"].min()) if len(similar) else low
            dearest = float(similar[similar["year"] >= year - 3]["price_lakh"].max()) if len(similar) else high
            span = max(dearest - cheapest, 1e-6)
            pos = min(max((price - cheapest) / span, 0), 1) * 100

            st.markdown(
                f"""
                <div class="price-card">
                  <div class="label">Estimated resale value</div>
                  <div class="value">₹ {price:.2f} L</div>
                  <div class="range">Likely range ₹ {low:.2f} – {high:.2f} lakh</div>
                  <div class="bar"><span style="left:{max(pos-3,0):.0f}%; width:6%"></span></div>
                  <div class="bar-labels"><span>₹ {cheapest:.1f} L</span>
                    <span>similar {model_name}s</span><span>₹ {dearest:.1f} L</span></div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # what-if curve: same car, different ages
            ages = list(range(1, 16))
            curve = predict([build_row(a, km / max(age, 1) * a, *args[1:]) for a in ages])
            st.markdown("#### How this car loses value")
            st.area_chart(pd.DataFrame({"Price (lakh ₹)": curve}, index=pd.Index(ages, name="Car age (years)")),
                          color="#22d3ee", height=210)

            k1, k2 = st.columns(2)
            k1.markdown(
                f'<div class="stat"><div class="t">Value after 1 more year</div>'
                f'<div class="n">₹ {float(predict([build_row(age + 1, km + 10000, *args[1:]) ])[0]):.2f} L</div></div>',
                unsafe_allow_html=True)
            k2.markdown(
                f'<div class="stat"><div class="t">Value after 3 more years</div>'
                f'<div class="n">₹ {float(predict([build_row(age + 3, km + 30000, *args[1:]) ])[0]):.2f} L</div></div>',
                unsafe_allow_html=True)
        else:
            st.info("Fill in the car details and press **Estimate price**.")

    if go or "last" in st.session_state:
        st.markdown("#### Similar cars in the dataset")
        comp = similar[similar["fuel"] == fuel].copy()
        # closest matches first: nearest in year, then nearest in kilometres driven
        comp["_dist"] = (comp["year"] - year).abs() * 50000 + (comp["km_driven"] - km).abs()
        comp = comp.sort_values("_dist").head(8)
        if len(comp):
            st.dataframe(
                comp[["brand", "model", "year", "km_driven", "fuel", "transmission", "owner", "city", "price_lakh"]]
                .rename(columns={"km_driven": "km", "price_lakh": "price (lakh ₹)"}),
                width="stretch", hide_index=True)
        else:
            st.caption("No close matches in the dataset for this combination.")

# ------------------------------------------------------------------ insights tab
with tab_insights:
    st.subheader("How well does the model work?")
    s1, s2, s3, s4 = st.columns(4)
    for col, label, val in [
        (s1, "Average error (MAE)", f"₹ {tm['MAE (lakh)']:.2f} L" if tm else "–"),
        (s2, "R² score", f"{tm['R2']:.3f}" if tm else "–"),
        (s3, "Avg % error (MAPE)", f"{tm['MAPE (%)']:.1f}%" if tm else "–"),
        (s4, "Test cars", f"{summary.get('test_rows', '–'):,}" if summary else "–"),
    ]:
        col.markdown(f'<div class="stat"><div class="t">{label}</div><div class="n">{val}</div></div>',
                     unsafe_allow_html=True)
    st.write("")

    metrics_path = find("outputs", "metrics.csv")
    if metrics_path.exists():
        st.markdown("#### Models compared")
        st.dataframe(pd.read_csv(metrics_path).round(3), width="stretch", hide_index=True)

    plots = [("model_comparison.png", "Model comparison"), ("pred_vs_actual.png", "Predicted vs actual"),
             ("feature_importance.png", "What drives the price"), ("error_distribution.png", "Error distribution")]
    cols = st.columns(2)
    for i, (f, cap) in enumerate(plots):
        p = find("outputs", f)
        if p.exists():
            cols[i % 2].image(str(p), caption=cap, width="stretch")

# ------------------------------------------------------------------ about tab
with tab_about:
    st.subheader("About this project")
    st.markdown(
        """
        **AutoWorth** estimates the resale price of a used car from its age, kilometres driven,
        brand and model, fuel, transmission, ownership history, city and engine details.

        **How it works**
        1. Data is cleaned (duplicates, impossible odometer readings and prices removed).
        2. Features are engineered: car age, kilometres per year, one-hot encoded categories.
        3. Price is modelled on a log scale because prices are right-skewed.
        4. Ridge, Random Forest, XGBoost and LightGBM are compared with cross-validation; the best is tuned.

        **Please note:** the demo dataset is **synthetic**. Estimates are illustrative and will be
        replaced by real market-based estimates once the model is retrained on real listings.
        """
    )
