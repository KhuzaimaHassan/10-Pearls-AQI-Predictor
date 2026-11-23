---
title: Pearls AQI Predictor
emoji: 🌍
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 8501
pinned: false
license: mit
---

#  Pearls AQI Predictor - Karachi Air Quality Forecast

An experimental Streamlit dashboard and scheduled ML pipeline for exploring Karachi air-quality forecasts. The repository includes hourly Open-Meteo ingestion, Hopsworks feature storage, and scheduled model training; these workflows depend on external credentials and successful provider runs.

[![Hugging Face Space](https://img.shields.io/badge/Hugging%20Face-Space-yellow)](https://huggingface.co/spaces/KhuzaimaHassan/10pearls-aqi-predictor)

---

##  **Project Overview**

This project provides **3-day ahead AQI predictions** for Karachi using machine learning models trained on historical weather and air quality data. The entire pipeline is fully automated using:

- **Hopsworks:** Feature Store & Model Registry
- **GitHub Actions:** Automated hourly data fetching & daily model training
- **Hugging Face Space:** Interactive dashboard deployment

---

##  **Features**

###  **Automated MLOps Pipeline**
- **Hourly Data Fetch**: `src/incremental_fetch.py` checks the latest Hopsworks timestamp, requests data from Open-Meteo, and uploads newly available hourly records
- **Daily Model Training**: `src/train_model.py` trains Random Forest, XGBoost, LightGBM and CatBoost models, plus a simple averaged ensemble, across 24/48/72-hour horizons (15 artifacts total)
- **Hopsworks Integration**: Feature Store & Model Registry

###  **Dashboard**
- **Forecasts**: 24/48/72-hour AQI estimates from the loaded model artifacts and available features
- **Historical Trends**: Interactive charts with Plotly
- **Model Performance**: R², MAE, RMSE metrics
- **Live Updates**: Data & models from Hopsworks

###  **Machine Learning**
- **Algorithms**: Random Forest, XGBoost, LightGBM, CatBoost, plus an averaged ensemble
- **Features**: Lag features, rolling statistics, wind components, temporal encodings
- **Optimization**: Optuna code exists, but `USE_BAYESIAN_OPT` is currently `False`; the scheduled run uses the configured base parameters

---

##  **Quick Start**

### **Prerequisites**
- Python 3.10+
- Hopsworks account ([Sign up free](https://www.hopsworks.ai/))
- GitHub account (for automation)

### **Installation**

1. **Clone Repository**
```bash
git clone https://github.com/KhuzaimaHassan/10-Pearls-AQI-Predictor.git
cd 10-Pearls-AQI-Predictor
```

2. **Install Dependencies**
```bash
pip install -r requirements.txt
```

3. **Configure Environment**
Create `.env` file:
```env
HOPSWORKS_API_KEY=your_api_key_here
HOPSWORKS_PROJECT_NAME=your_project_name
```

4. **Run Pipeline**
```bash
# Fetch historical data (requires configured Hopsworks access for upload)
python src/fetch_data.py

# Generate and upload the initial feature set
python src/feature_pipeline.py

# WARNING: training removes existing local *.pkl model files before starting
python src/train_model.py

# Run dashboard
streamlit run app.py
```

---

##  **Project Structure**

```
Pearls_AQI/
├── .github/workflows/       # GitHub Actions automation
│   ├── feature_pipeline.yml # Hourly incremental fetch & upload
│   └── train_model.yml      # Daily initial fetch, feature generation, and training
├── src/
│   ├── config.py           # Configuration
│   ├── fetch_data.py       # Historical data acquisition
│   ├── incremental_fetch.py # Hourly incremental acquisition & upload
│   ├── feature_pipeline.py # Initial feature engineering
│   └── train_model.py      # Model training
├── app.py                  # Streamlit dashboard
├── requirements.txt        # Python dependencies
├── packages.txt           # System packages (for Streamlit Cloud)
└── README.md
```

---

##  **MLOps Pipeline**

### **Hourly Workflow** (GitHub Actions)
1. Fetch latest AQI & weather data (Open-Meteo API)
2. Create temporal, interaction, wind, lag, and rolling feature columns; incremental rows initialize lag/rolling fields as missing
3. Upload to Hopsworks Feature Store
4. Run every hour via GitHub Actions

### **Daily Workflow** (GitHub Actions)
1. Run daily at **02:00 UTC (07:00 Pakistan time)**, subject to GitHub Actions schedule availability
2. Fetch historical data and regenerate features
3. Train the configured model families for 24-, 48-, and 72-hour horizons with time-series cross-validation
4. Bayesian Optuna tuning is currently disabled in `src/train_model.py`
5. Upload model artifacts through `upload_all_models.py` to the Hopsworks registry

### **Dashboard** (Hugging Face space)
1. Load features from Hopsworks Feature Store
2. Download models from Hopsworks Model Registry
3. Generate 3-day predictions
4. Display interactive visualizations
5. Auto-updates with new data & models

---

##  **Data Sources**

- **Open-Meteo API**: Historical & forecast weather data
  - Temperature, humidity, wind speed, precipitation
  - PM2.5, PM10, US AQI
  - Location: Karachi (24.8607°N, 67.0011°E)

---

##  **Model Performance**

The README does not publish performance estimates because the repository does not provide a linked, reproducible current evaluation report establishing the listed R² and MAE ranges. Training code computes time-series CV and a chronological 70/30 holdout for each model run; inspect that run's output before making performance claims.

---

##  **Technologies Used**

| Category | Technologies |
|----------|-------------|
| **ML Frameworks** | XGBoost, LightGBM, CatBoost, Scikit-learn |
| **Feature Store** | Hopsworks |
| **Orchestration** | GitHub Actions |
| **Dashboard** | Streamlit, Plotly |
| **Data** | Pandas, NumPy |
| **Optimization** | Optuna |

---

##  **Configuration**

### **GitHub Secrets** (for automation)
Add these to your repository settings:
```
HOPSWORKS_API_KEY=<your_hopsworks_api_key>
HOPSWORKS_PROJECT_NAME=<your_project_name>
```

### **Hugging Face App Secrets**
Add to  Hugging Face space settings:
```toml
HOPSWORKS_API_KEY='<your_hopsworks_api_key>'
HOPSWORKS_PROJECT_NAME='<your_project_name>'
```

---

##  **Contributing**

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

---

##  **License**

This project is licensed under the MIT License.

---

##  **Author**

**Khuzaima Hassan**
- GitHub: [@KhuzaimaHassan](https://github.com/KhuzaimaHassan)

---

##  **Acknowledgments**

- **Open-Meteo** for free weather & AQI data API
- **Hopsworks** for Feature Store & Model Registry
- **Hugging Face** for easy dashboard deployment
- **10 Pearls** for project inspiration

---

##  **Contact**

For questions or feedback, please open an issue on GitHub.
