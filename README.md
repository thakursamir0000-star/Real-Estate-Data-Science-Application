# real-estate-analytics-and-recommendation-system

An **end-to-end Data Science project** that analyzes real estate data, predicts property prices, and recommends properties based on user preferences.

This project demonstrates the complete data science pipeline including **data collection, data cleaning, exploratory analysis, feature engineering, machine learning modeling, and deployment of a web application**.

---

# 📌 Project Overview

The goal of this project is to apply **data science and machine learning techniques** to the real estate domain in order to generate insights, predict property prices, and build a recommendation system for users.

The application provides:

* 📊 Real estate analytics dashboard
* 💰 Property price prediction system
* 🏡 Property recommendation system
* 🌐 Interactive web application using Streamlit
* ☁️ Deployment on AWS

---

# 🚀 Features

## 1️⃣ Price Prediction

A machine learning model predicts property prices based on user inputs such as:

* Location
* Area (sq.ft.)
* Number of bedrooms
* Furnishing status
* Property age
* Amenities

The system returns an **estimated property price**.

---

## 2️⃣ Analytics Dashboard

The analytics module provides insights into the real estate market through visualizations such as:

* Geographic price distribution maps
* Scatter plots and box plots
* Pie charts for property distribution
* Word clouds for amenities
* Market trend visualizations

---

## 3️⃣ Property Recommendation System

Three recommendation models provide personalized property suggestions based on:

* Top facilities
* Price preferences
* Location advantages

Users receive **tailored property recommendations**.

---

# 🧠 Machine Learning Models Used

Several regression models were tested and compared:

* Linear Regression
* Support Vector Regression (SVR)
* Random Forest Regressor
* Gradient Boosting Regressor
* Decision Tree Regressor
* K-Nearest Neighbors Regressor
* Ridge Regression
* LASSO Regression
* ElasticNet Regression
* Multi-layer Perceptron (MLP)

The best-performing model was selected based on evaluation metrics.

---

# ⚙️ Project Workflow

## 1. Data Gathering

* Real estate data was scraped from **99acres**
* Additional datasets from property listing websites were explored

## 2. Data Cleaning

* Handling missing values
* Standardizing features
* Merging datasets

## 3. Feature Engineering

New features were created to improve model performance:

* Area with type
* Possession age
* Furnishing details
* Additional room indicators
* Luxury score

## 4. Exploratory Data Analysis (EDA)

EDA was performed using:

* Pandas
* Matplotlib
* Seaborn
* Pandas Profiling

## 5. Outlier Detection

Outliers were detected and removed to improve model robustness.

## 6. Feature Selection

Various techniques were used:

* Correlation analysis
* Random Forest feature importance
* Gradient Boosting importance
* Permutation importance
* LASSO
* Recursive Feature Elimination (RFE)
* SHAP (Explainable AI)

## 7. Model Building

* Training and evaluation of multiple regression models
* Creation of a price prediction pipeline

## 8. Application Development

An interactive **Streamlit web application** was built to allow users to interact with the model.

## 9. Deployment

The full application was deployed on **Amazon Web Services (AWS)**.

---


# 🛠 Tech Stack

### Programming Language

Python

### Libraries

* Pandas
* NumPy
* Scikit-learn
* Matplotlib
* Seaborn
* SHAP

### Web Application

Streamlit

### Deployment

AWS

---

# 📊 Model Evaluation Metrics

The models were evaluated using:

* R² Score
* Mean Absolute Error (MAE)
* Mean Squared Error (MSE)
* Root Mean Squared Error (RMSE)

---

-

# 🎯 Project Highlights

* End-to-end Data Science workflow
* Multiple machine learning models comparison
* Feature engineering and explainable AI
* Interactive Streamlit dashboard
* Real-world deployment on AWS

---

# 👨‍💻 Author

**Samir Thakur**

MSc AI & ML Indian Institute of Information Technology
Interested in Machine Learning, Deep Learning, and AI Applications
