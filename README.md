<p align="center">
  <h1 align="center">🏘️ Gurgaon Real Estate: Price Prediction & Recommendations</h1>
  <p align="center">
    An end-to-end data science project on ~3,500 Gurgaon property listings: notebooks for cleaning and feature engineering, a cross-validated price model with calibrated price ranges and per-prediction explanations, two recommenders, and a Streamlit app.
  </p>
</p>

<p align="center">
  <a href="https://real-estate-data-science-application-3gbgmpmtnuetxhuvrracam.streamlit.app/">
    <img src="https://static.streamlit.io/badges/streamlit_badge_black_white.svg" alt="Open in Streamlit" />
  </a>
  <a href="https://github.com/thakursamir0000-star/Real-Estate-Data-Science-Application/actions/workflows/ci.yml">
    <img src="https://github.com/thakursamir0000-star/Real-Estate-Data-Science-Application/actions/workflows/ci.yml/badge.svg" alt="CI" />
  </a>
  <img src="https://img.shields.io/badge/python-3.11+-blue.svg" alt="Python 3.11+" />
  <img src="https://img.shields.io/badge/scikit--learn-1.8.0-orange.svg" alt="scikit-learn" />
  <img src="https://img.shields.io/badge/XGBoost-3.2-green.svg" alt="XGBoost" />
</p>

---

## 🔍 Overview

| Module | What it does |
|---|---|
| 💰 **Price Predictor** | An estimate in ₹ crore, an **80% price range** that really contains the price ~80% of the time, and **what drives this price** (e.g. "sector 54: +70% vs a typical sector") |
| 📊 **Analytics Dashboard** | A **map of price per sq ft by sector**, the most expensive sectors, price vs area and bedrooms, and the model comparison |
| 🏡 **Recommend Properties** | **Similar listings** (same type, close in price, size, bedrooms, bathrooms) and **similar housing societies** by shared facilities, price level and nearby places, with the reasons for each match |

The data are ~3,500 listings scraped from 99acres (flats and independent houses across 104 sectors). Prices are **listing** prices, not transaction prices.

---

## 📊 Results

All models were scored with the same **10-fold cross-validation** on the 3,520 de-duplicated listings (`python -m src.models.train_models`; full numbers in [`models/metrics.json`](models/metrics.json)). Models learn `log1p(price)`; errors are on the ₹ crore scale.

| Model | MAE (₹ Cr) | Median error | MAPE | R² (log price) |
|---|---|---|---|---|
| Ridge (linear baseline) | 0.659 | 17.9% | 27.3% | 0.857 |
| Random Forest — the original app's model and encoding | 0.528 | 13.5% | 22.4% | 0.883 |
| Random Forest (target-encoded sectors) | **0.522** | 14.0% | 22.2% | 0.889 |
| Extra Trees | 0.537 | 14.0% | 22.4% | 0.886 |
| **XGBoost — deployed** | 0.529 | 14.8% | 22.5% | **0.892** |
| LightGBM | 0.547 | 15.3% | 23.0% | 0.886 |

**How the deployed model was chosen.** The tree models are within ~4% of each other, and fold-to-fold noise (standard deviation of MAE ≈ ±0.05 Cr) is about ten times larger than the gaps between the top three. So instead of taking the lowest number, the **one-standard-error rule** picks the cheapest model within one standard error of the best MAE: XGBoost (0.529 vs a 0.539 threshold), a **0.7 MB** file instead of the Random Forest's 20.8 MB (the original pickle was 29.5 MB).

**Calibrated price ranges.** Ranges are split-conformal intervals built from the out-of-fold residuals and calibrated **separately for flats and houses**, because house prices are about twice as hard to pin down. Coverage was checked honestly (each fold's width calibrated on the other nine folds):

| Range | Width (flats) | Width (houses) | Real prices inside — flats / houses / all |
|---|---|---|---|
| 80% | −15% … +18% | −28% … +39% | 80.0% / 80.2% / 80.0% |
| 90% | −20% … +26% | −38% … +60% | 89.9% / 90.0% / 89.9% |

**What didn't help, and why it's not used:**
- *De-duplication* (34 exact duplicate rows) is done for correctness, but barely changed the scores here.
- *Forcing price to rise with area* (an XGBoost monotonic constraint) removed occasional tiny dips but made MAE **4% worse** (0.529 → 0.551). Instead, the app reports feature effects under ±5% as "little clear effect" rather than as drivers.
- The original model was already close to what these 12 features allow; the gains of this version are honest evaluation, uncertainty and explanations rather than a lower error.

---

## ✨ Features

### 💰 Price prediction with a range and explanations
- 12 inputs: property type, sector, bedrooms, bathrooms, balconies, age/possession, built-up area, servant room, store room, furnishing (unfurnished / semi-furnished / furnished), luxury level, floor.
- Output: estimate, the 80% range for that property type, price per sq ft, and the top drivers.
- **What drives this price** is a what-if explanation (`PricePredictor.explain`): each feature is set to its most common value for that property type and the change in the estimate is shown. It works for any model and needs no extra library; it's approximate because features interact.

### 📊 Analytics dashboard
- Median price per sq ft by sector on a map (sectors with ≥3 listings and known coordinates; the coordinate file's mis-encoded degree signs are parsed in `load_sector_coordinates`).
- Most expensive sectors, price vs area (log scale), price by bedrooms, and the cross-validated model comparison.

### 🏡 Recommendations
1. **Similar listings** — nearest neighbours among the same property type on *standardized* log price, log area, bedrooms and bathrooms (the original used raw values, so square feet swamped everything else), optionally within the same sector.
2. **Similar housing societies** — 247 societies from `data/raw/appartments.csv`, blending facility overlap (52%), price level (34%) and shared nearby places (14%), the weights from `notebooks/recommender-system.ipynb`; each match lists the shared facilities and nearby places.

---

## 🏗 Project Structure

```
Real-Estate-Data-Science-Application/
├── app/
│   ├── app.py                      # Home page with headline metrics
│   └── pages/
│       ├── 1_Price_Predictor.py    # Estimate + range + drivers
│       ├── 2_Analytics_Dashboard.py
│       └── 3_Recommend_Properties.py
├── src/
│   ├── data/make_dataset.py        # Load, validate, de-duplicate, readable labels, sector coordinates
│   ├── models/train_models.py      # 10-fold CV of 6 models, model selection, conformal ranges, model card
│   ├── models/predict.py           # PricePredictor: predict() and explain()
│   └── recommend.py                # similar_listings, SocietyRecommender
├── models/
│   ├── price_model.joblib          # Deployed pipeline (preprocessing + XGBoost)
│   └── metrics.json                # Model card: CV metrics, selection, ranges, library versions
├── data/
│   ├── raw/                        # Scraped listings and societies
│   └── processed/                  # Modelling table + sector coordinates
├── notebooks/                      # EDA, cleaning, feature engineering, selection, experiments
├── tests/                          # pytest suite
└── .github/workflows/ci.yml        # Lint, tests, re-training reproducibility check
```

### Data pipeline

```
99acres scraping → data/raw (flats, houses, societies)
  → notebooks: cleaning, merging, missing values, outliers, feature engineering
    (luxury score, floor category, furnishing clusters), feature selection
  → data/processed/gurgaon_properties_post_feature_selection_v2.csv
  → src/data/make_dataset.py: schema checks, duplicates, labels
  → src/models/train_models.py → models/price_model.joblib + metrics.json
```

| Notebook | Purpose |
|---|---|
| `data-preprocessing-flats.ipynb`, `data-preprocessing-houses.ipynb`, `merge-flats-and-house.ipynb`, `data-preprocessing-level-2.ipynb` | Clean and merge listings |
| `missing-value-imputation.ipynb`, `outlier-treatment.ipynb` | Missing values and outliers |
| `eda-univariate-analysis.ipynb`, `eda-multivariate-analysis.ipynb`, `eda-pandas-profiling.ipynb`, `data-visualization.ipynb` | Exploration |
| `feature-engineering.ipynb`, `feature-selection.ipynb`, `feature-selection-and-feature-engineering.ipynb` | Features |
| `baseline model.ipynb`, `model-selection.ipynb` | Model experiments |
| `insights-module.ipynb`, `recommender-system.ipynb` | Insights and the society recommender's design |

---

## 🚀 Setup

```bash
git clone https://github.com/thakursamir0000-star/Real-Estate-Data-Science-Application.git
cd Real-Estate-Data-Science-Application
python -m venv venv
venv\Scripts\activate            # Windows  (source venv/bin/activate on macOS/Linux)
pip install -r requirements.txt

streamlit run app/app.py          # http://localhost:8501
```

Retrain and refresh the model card (about a minute):

```bash
python -m src.models.train_models          # 10-fold CV of all models, saves models/
python -m src.models.train_models --quick  # 3 folds, 2 small models (smoke test)
```

`models/price_model.joblib` was trained with scikit-learn 1.8.0 and XGBoost 3.2.0, which `requirements.txt` pins exactly; retrain before changing them.

---

## 🧪 Testing

```bash
pip install -r requirements-dev.txt
ruff check .
pytest
```

The tests cover data validation and coordinate parsing, the conformal-interval maths and the model-selection rule, a quick end-to-end training run, predictions (ranges contain the estimate, houses get wider ranges, prime sectors cost more) and explanations, both recommenders, and every app page (including clicking **Predict**). CI also **re-trains the full model from scratch** and fails if any model's MAE drifts more than 3% from the committed model card, so the published numbers stay reproducible.

---

## ⚠️ Limitations

- **Listing prices, one snapshot.** Asking prices from one scrape of 99acres, not transaction prices, and they age.
- **Gurgaon only**; sectors with few listings get less reliable estimates (their ranges are not widened separately).
- **Explanations are approximate** what-if effects, not causal effects.
- **Society data** covers 247 societies and their listed facilities; some have no published prices.

---

## 👨‍💻 Author

**Samir Thakur** — MSc in AI & ML, Indian Institute of Information Technology

[![GitHub](https://img.shields.io/badge/GitHub-thakursamir0000--star-181717?logo=github)](https://github.com/thakursamir0000-star)
