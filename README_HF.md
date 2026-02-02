---
title: 10 Pearls AQI Predictor
emoji: 🌍
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
---

# 🌍 Pearls AQI Predictor - Karachi Air Quality Forecast

**3-Day Air Quality Index predictions for Karachi, Pakistan using MLOps pipeline with automated hourly data updates and daily model retraining.**

## 📊 Features

- **Real-time 3-Day Predictions**: AQI forecast for Karachi
- **Live Data**: Fresh hourly updates from Hopsworks Feature Store
- **ML Models**: 15 optimized models (XGBoost, LightGBM, CatBoost, RF, Linear)
- **Interactive Dashboard**: Plotly visualizations
- **Automated Pipeline**: GitHub Actions for data & model updates

## 🚀 Tech Stack

- **ML Frameworks**: XGBoost, LightGBM, CatBoost, Scikit-learn
- **Feature Store**: Hopsworks
- **Dashboard**: Streamlit
- **Orchestration**: GitHub Actions
- **Data Source**: Open-Meteo API

## 📝 About

This project demonstrates a complete MLOps pipeline for air quality prediction:
- Hourly data fetching and feature engineering
- Daily model training with Optuna hyperparameter tuning
- Real-time predictions from Hopsworks Model Registry

**Created for 10 Pearls internship project** | [GitHub](https://github.com/KhuzaimaHassan/10-Pearls-AQI-Predictor)
