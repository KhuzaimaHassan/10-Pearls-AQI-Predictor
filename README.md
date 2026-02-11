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

**3-Day Air Quality Index predictions for Karachi, Pakistan using MLOps pipeline with automated hourly data updates and daily model retraining.**

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
- **Hourly Data Fetch**: Fresh AQI & weather data from Open-Meteo API
- **Hourly Feature Engineering**: 32 features with lag, rolling windows, temporal encodings
- **Daily Model Training**: 15 models (5 algorithms × 3 days) with Optuna hyperparameter tuning
- **Hopsworks Integration**: Feature Store & Model Registry

###  **Dashboard**
- **Real-time Predictions**: 3-day AQI forecast
- **Historical Trends**: Interactive charts with Plotly
- **Model Performance**: R², MAE, RMSE metrics
- **Live Updates**: Data & models from Hopsworks

###  **Machine Learning**
- **Algorithms**: XGBoost, LightGBM, CatBoost, Random Forest, Linear Regression
- **Features**: Lag features, rolling statistics, wind components, temporal encodings
- **Optimization**: Optuna for hyperparameter tuning

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
# Fetch data
python src/fetch_data.py

# Generate features
python src/feature_pipeline.py

# Train models
python src/train_model.py

# Run dashboard
streamlit run app.py
```

---

##  **Project Structure**

```
Pearls_AQI/
├── .github/workflows/       # GitHub Actions automation
│   ├── feature_pipeline.yml # Hourly data fetch & upload
│   └── train_model.yml      # Daily model training
├── notebooks/               # EDA & analysis notebooks
├── src/
│   ├── config.py           # Configuration
│   ├── fetch_data.py       # Data acquisition
│   ├── feature_pipeline.py # Feature engineering
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
2. Generate 32 engineered features
3. Upload to Hopsworks Feature Store
4. Run every hour via GitHub Actions

### **Daily Workflow** (GitHub Actions)
1. Download features from Hopsworks
2. Train 15 models (5 algorithms × 3 forecast days)
3. Tune hyperparameters with Optuna
4. Upload models to Hopsworks Model Registry
5. Run daily at midnight UTC

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

Current models achieve:
- **Day 1**: R² ~0.85-0.90, MAE ~15-20
- **Day 2**: R² ~0.75-0.82, MAE ~20-28  
- **Day 3**: R² ~0.65-0.75, MAE ~25-35

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
