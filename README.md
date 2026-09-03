<p align="center">
  <h1 align="center">🏘️ Real Estate Analytics & Recommendation System</h1>
  <p align="center">
    An end-to-end Data Science application for analyzing Gurgaon real estate data, predicting property prices, and recommending properties based on user preferences.
  </p>
</p>

<p align="center">
  <a href="https://real-estate-data-science-application-3gbgmpmtnuetxhuvrracam.streamlit.app/">
    <img src="https://static.streamlit.io/badges/streamlit_badge_black_white.svg" alt="Open in Streamlit" />
  </a>
  <img src="https://img.shields.io/badge/python-3.9+-blue.svg" alt="Python 3.9+" />
  <img src="https://img.shields.io/badge/license-MIT-green.svg" alt="MIT License" />
  <img src="https://img.shields.io/badge/scikit--learn-1.8.0-orange.svg" alt="scikit-learn" />
  <img src="https://img.shields.io/badge/streamlit-latest-red.svg" alt="Streamlit" />
</p>

<p align="center">
  🌍 <strong>Live Demo:</strong> <a href="https://real-estate-data-science-application-3gbgmpmtnuetxhuvrracam.streamlit.app/">Click here to view the deployed application</a>
</p>

---

## 📌 Table of Contents

- [Overview](#-overview)
- [Features](#-features)
- [Project Architecture](#-project-architecture)
- [Tech Stack](#️-tech-stack)
- [Installation & Setup](#-installation--setup)
- [Usage](#-usage)
- [Data Pipeline](#-data-pipeline)
- [Machine Learning Models](#-machine-learning-models)
- [Model Evaluation](#-model-evaluation)
- [Project Workflow](#️-project-workflow)
- [Notebooks](#-notebooks)
- [Deployment](#-deployment)
- [Contributing](#-contributing)
- [License](#-license)
- [Author](#-author)

---

## 🔍 Overview

This project applies **data science and machine learning techniques** to the Gurgaon real estate market. It covers the entire ML lifecycle — from web scraping and data cleaning to model training, evaluation, and deployment as an interactive web application.

**What can you do with this app?**

| Module | Description |
|--------|-------------|
| 💰 **Price Predictor** | Estimate property prices using a trained Random Forest pipeline |
| 📊 **Analytics Dashboard** | Explore interactive charts — pie charts, bar plots, scatter plots, box plots |
| 🏡 **Recommender System** | Get similar property suggestions using content-based filtering (Euclidean distance) |

---

## ✨ Features

### 💰 Price Prediction
Predicts property prices (in Crores) based on 12 input features:
- Property type, sector, bedrooms, bathrooms, balconies
- Built-up area (sq ft), servant room, store room
- Age/possession status, furnishing type
- Luxury category, floor category

The model uses a `log1p` transformation on the target variable and returns predictions via `expm1` for accurate price estimation.

### 📊 Analytics Dashboard
Interactive Plotly-powered visualizations including:
- **Pie chart** — Property type distribution
- **Bar chart** — Top 10 sectors by average price
- **Scatter plot** — Price vs built-up area (colored by property type)
- **Box plot** — Price distribution across bedroom counts
- **Histogram** — Property count by luxury category

### 🏡 Property Recommendation
Two recommendation approaches:
1. **Filter-based** — Narrow down properties by sector, price range, and minimum bedrooms
2. **Content-based similarity** — Select a property and find the 5 most similar ones using Euclidean distance on `[price, bedRoom, bathroom, built_up_area]`

---

## 🏗 Project Architecture

```
Real-Estate-Data-Science-Application/
│
├── app/                          # Streamlit web application
│   ├── app.py                    # Main entry point (home page)
│   └── pages/
│       ├── 1_Price_Predictor.py  # Price prediction module
│       ├── 2_Analytics_Dashboard.py  # Data visualization module
│       └── 3_Recommend_Properties.py # Recommendation engine
│
├── data/
│   ├── raw/                      # Original scraped datasets
│   │   ├── gurgaon_properties.csv
│   │   ├── flats.csv
│   │   ├── houses.csv
│   │   └── appartments.csv
│   └── processed/                # Cleaned & feature-engineered data
│       ├── gurgaon_properties_post_feature_selection_v2.csv
│       └── latlong.csv
│
├── models/                       # Serialized model artifacts
│   ├── pipeline.pkl              # Trained sklearn Pipeline (preprocessor + model)
│   └── df.pkl                    # Feature DataFrame for UI dropdowns
│
├── notebooks/                    # Jupyter notebooks (full EDA & experimentation)
│   ├── data-preprocessing-flats.ipynb
│   ├── data-preprocessing-houses.ipynb
│   ├── data-preprocessing-level-2.ipynb
│   ├── merge-flats-and-house.ipynb
│   ├── missing-value-imputation.ipynb
│   ├── eda-univariate-analysis.ipynb
│   ├── eda-multivariate-analysis.ipynb
│   ├── eda-pandas-profiling.ipynb
│   ├── data-visualization.ipynb
│   ├── feature-engineering.ipynb
│   ├── feature-selection.ipynb
│   ├── feature-selection-and-feature-engineering.ipynb
│   ├── outlier-treatment.ipynb
│   ├── baseline model.ipynb
│   ├── model-selection.ipynb
│   ├── insights-module.ipynb
│   └── recommender-system.ipynb
│
├── src/                          # Reusable Python modules
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   └── make_dataset.py       # Data loading & cleaning utilities
│   ├── features/
│   │   ├── __init__.py
│   │   └── build_features.py     # ColumnTransformer (StandardScaler + OrdinalEncoder)
│   └── models/
│       ├── __init__.py
│       └── train_models.py       # Model training & serialization script
│
├── model_comparison.py           # 10-fold CV comparison of 10 regression models
├── extract_metrics.py            # Dataset statistics extraction utility
├── requirements.txt              # Python dependencies
├── setup.py                      # Package configuration
├── .gitignore
└── README.md
```

---

## 🛠️ Tech Stack

| Category | Technologies |
|----------|-------------|
| **Language** | Python 3.9+ |
| **ML / Data** | scikit-learn 1.8.0, Pandas, NumPy |
| **Visualization** | Plotly, Matplotlib, Seaborn, WordCloud |
| **Encoding** | category_encoders, OrdinalEncoder |
| **Explainability** | SHAP (in notebooks) |
| **Web Framework** | Streamlit |
| **Deployment** | AWS, Streamlit Cloud |
| **Profiling** | Pandas Profiling |

---

## 🚀 Installation & Setup

### Prerequisites
- Python 3.9 or higher
- pip package manager

### 1. Clone the repository

```bash
git clone https://github.com/thakursamir0000-star/Real-Estate-Data-Science-Application.git
cd Real-Estate-Data-Science-Application
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Train the model (if `models/` directory is empty)

```bash
python src/models/train_models.py
```

This will:
- Load the processed dataset from `data/processed/`
- Build a `StandardScaler + OrdinalEncoder → RandomForestRegressor` pipeline
- Save `pipeline.pkl` and `df.pkl` to the `models/` directory

### 5. Run the Streamlit app

```bash
cd app
streamlit run app.py
```

The app will open at `http://localhost:8501`.

---

## 💡 Usage

### Price Prediction
1. Navigate to **💰 Price Predictor** in the sidebar
2. Fill in the 12 property features (type, sector, area, bedrooms, etc.)
3. Click **"Predict Price"** to get the estimated price in Crores (₹)

### Analytics Dashboard
1. Navigate to **📊 Analytics Dashboard**
2. Explore interactive charts showing market distributions, price trends, and sector comparisons

### Property Recommendations
1. Navigate to **🏡 Recommend Properties**
2. Use the sidebar filters (sector, price range, bedrooms) to find matching properties
3. Or select a specific property and click **"Find Similar Properties"** for content-based recommendations

---

## 🔄 Data Pipeline

```
99acres Web Scraping
        │
        ▼
┌─────────────────────┐
│   Raw Data (CSV)    │  flats.csv, houses.csv, appartments.csv
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  Data Preprocessing │  Cleaning, merging, missing value imputation
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  Feature Engineering│  Luxury score, area type, possession age, etc.
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  Outlier Treatment  │  Statistical detection & removal
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  Feature Selection  │  Correlation, RF importance, SHAP, RFE, LASSO
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  Processed Dataset  │  gurgaon_properties_post_feature_selection_v2.csv
└─────────────────────┘
```

---

## 🧠 Machine Learning Models

Ten regression models were benchmarked using **10-fold cross-validation** on `log1p(price)`:

| # | Model | Description |
|---|-------|-------------|
| 1 | Linear Regression | Baseline ordinary least squares |
| 2 | Ridge Regression | L2 regularization (α=1.0) |
| 3 | Lasso Regression | L1 regularization (α=0.1) |
| 4 | ElasticNet | Combined L1+L2 regularization (α=0.1) |
| 5 | SVR | Support Vector Regression (RBF kernel) |
| 6 | Decision Tree | Single decision tree regressor |
| 7 | **Random Forest** ⭐ | Ensemble of 100 trees — **selected for production** |
| 8 | Gradient Boosting | Sequential boosting regressor |
| 9 | K-Nearest Neighbors | Distance-based regression (k=5) |
| 10 | MLP Regressor | Neural network with 100 hidden units |

> **Production model:** Random Forest Regressor was selected as the final model and is served through the Streamlit application.

### Model Pipeline

```
Input Features (12)
       │
       ▼
┌──────────────────────────────────┐
│        ColumnTransformer         │
│  ┌────────────┬────────────────┐ │
│  │ Numerical  │  Categorical   │ │
│  │ Standard   │  Ordinal       │ │
│  │ Scaler     │  Encoder       │ │
│  └────────────┴────────────────┘ │
└──────────────┬───────────────────┘
               │
               ▼
┌──────────────────────────────────┐
│   Random Forest Regressor        │
│   (n_estimators=100)             │
└──────────────┬───────────────────┘
               │
               ▼
       log1p(price) → expm1 → ₹ Crores
```

**Numerical features** (5): `bedRoom`, `bathroom`, `built_up_area`, `servant room`, `store room`

**Categorical features** (7): `property_type`, `sector`, `balcony`, `agePossession`, `furnishing_type`, `luxury_category`, `floor_category`

---

## 📊 Model Evaluation

Models were evaluated using the following metrics:

| Metric | Description |
|--------|-------------|
| **R² Score** | Coefficient of determination (higher is better) |
| **MAE** | Mean Absolute Error in Crores (lower is better) |
| **MSE** | Mean Squared Error (lower is better) |
| **RMSE** | Root Mean Squared Error (lower is better) |

Run the model comparison yourself:

```bash
python model_comparison.py
```

This script performs 10-fold cross-validation on all 10 models and prints a ranked comparison table.

---

## ⚙️ Project Workflow

| Step | Description | Notebooks / Scripts |
|------|-------------|-------------------|
| 1. **Data Gathering** | Scraped property listings from 99acres | — |
| 2. **Data Preprocessing** | Cleaned flats & houses separately, then merged | `data-preprocessing-flats.ipynb`, `data-preprocessing-houses.ipynb`, `merge-flats-and-house.ipynb` |
| 3. **Missing Values** | Imputed missing values using domain knowledge | `missing-value-imputation.ipynb` |
| 4. **EDA** | Univariate, multivariate analysis & profiling | `eda-univariate-analysis.ipynb`, `eda-multivariate-analysis.ipynb`, `eda-pandas-profiling.ipynb` |
| 5. **Feature Engineering** | Created luxury score, area type, possession age | `feature-engineering.ipynb` |
| 6. **Outlier Treatment** | Statistical detection and removal | `outlier-treatment.ipynb` |
| 7. **Feature Selection** | Correlation, RF importance, SHAP, RFE, LASSO | `feature-selection.ipynb`, `feature-selection-and-feature-engineering.ipynb` |
| 8. **Model Training** | Trained & compared 10 regression models | `model-selection.ipynb`, `baseline model.ipynb` |
| 9. **Visualization** | Built interactive data visualizations | `data-visualization.ipynb`, `insights-module.ipynb` |
| 10. **Recommendation** | Content-based property recommender | `recommender-system.ipynb` |
| 11. **Deployment** | Streamlit app deployed on AWS & Streamlit Cloud | `app/` |

---

## 📓 Notebooks

The `notebooks/` directory contains 17 Jupyter notebooks documenting every step of the data science process:

| Notebook | Purpose |
|----------|---------|
| `data-preprocessing-flats.ipynb` | Clean and preprocess flat listings |
| `data-preprocessing-houses.ipynb` | Clean and preprocess house listings |
| `data-preprocessing-level-2.ipynb` | Second-level data preprocessing |
| `merge-flats-and-house.ipynb` | Merge flat and house datasets |
| `missing-value-imputation.ipynb` | Handle missing values |
| `eda-univariate-analysis.ipynb` | Single-variable analysis |
| `eda-multivariate-analysis.ipynb` | Multi-variable relationships |
| `eda-pandas-profiling.ipynb` | Automated profiling report |
| `data-visualization.ipynb` | Comprehensive data visualizations |
| `feature-engineering.ipynb` | Create new features |
| `feature-selection.ipynb` | Select most important features |
| `feature-selection-and-feature-engineering.ipynb` | Combined selection & engineering |
| `outlier-treatment.ipynb` | Detect and remove outliers |
| `baseline model.ipynb` | Initial baseline model |
| `model-selection.ipynb` | Compare and select best model |
| `insights-module.ipynb` | Generate market insights |
| `recommender-system.ipynb` | Build recommendation engine |

---

## ☁️ Deployment

The application is deployed on:

- **Streamlit Cloud** — [Live Demo](https://real-estate-data-science-application-3gbgmpmtnuetxhuvrracam.streamlit.app/)
- **AWS** — Full production deployment

---

## 🤝 Contributing

Contributions are welcome! To contribute:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the **MIT License** — see the [setup.py](setup.py) for details.

---

## 👨‍💻 Author

**Samir Thakur**

MSc in AI & ML — Indian Institute of Information Technology

Interested in Machine Learning, Deep Learning, and AI Applications

[![GitHub](https://img.shields.io/badge/GitHub-thakursamir0000--star-181717?logo=github)](https://github.com/thakursamir0000-star)
