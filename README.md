# Pearls AQI Predictor

![Python](https://img.shields.io/badge/python-3.9%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)

A complete end-to-end machine learning pipeline for predicting Air Quality Index (AQI) in Karachi for the next 3 days using hourly weather data and advanced feature engineering.

## 🌟 Features

- **Automated Data Pipeline**: Hourly data fetching from Open-Meteo API
- **Feature Engineering**: Lag features, rolling statistics, cyclical encoding
- **Multi-Model Training**: Linear Regression, Random Forest, XGBoost
- **Dynamic Model Selection**: Automatically uses the best-performing model daily
- **Interactive Dashboard**: Streamlit app with real-time predictions and visualizations
- **CI/CD Automation**: GitHub Actions for continuous model improvement

## 🏗️ Architecture

```
Open-Meteo API → Feature Engineering → Hopsworks Feature Store
                                    ↓
                              Model Training → Hopsworks Model Registry
                                    ↓
                              Streamlit Dashboard
```

## 📋 Requirements

- Python 3.9 or higher
- Hopsworks account (free tier available at [hopsworks.ai](https://www.hopsworks.ai/))
- Git (for version control and CI/CD)

## 🚀 Quick Start

### 1. Clone the Repository

```bash
git clone <your-repo-url>
cd Pearls_AQI
```

### 2. Set Up Virtual Environment

**Windows (PowerShell):**
```powershell
python -m venv .venv
.\.venv\Scripts\activate
```

**Linux/Mac:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies

```powershell
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Configure Environment Variables

Create a `.env` file in the project root:

```env
HOPSWORKS_API_KEY=your_api_key_here
HOPSWORKS_PROJECT_NAME=your_project_name
```

**To get your Hopsworks API key:**
1. Sign up at [hopsworks.ai](https://www.hopsworks.ai/)
2. Create a new project
3. Go to Settings → API Keys → Generate new key

### 5. Run the Data Pipeline

**Fetch historical data:**
```powershell
python src/fetch_data.py
```

**Engineer features and upload to Hopsworks:**
```powershell
python src/feature_pipeline.py
```

**Train models:**
```powershell
python src/train_model.py
```

### 6. Launch the Dashboard

```powershell
streamlit run app.py
```

The app will open at `http://localhost:8501`

## 📊 Project Structure

```
Pearls_AQI/
├── .venv/                      # Virtual environment (not in Git)
├── .github/
│   └── workflows/
│       └── data_pipeline.yml   # GitHub Actions automation
├── data/
│   ├── raw/                    # Raw API data
│   └── processed/              # Engineered features
├── models/                     # Trained model artifacts
├── notebooks/                  # Jupyter notebooks (exploratory analysis)
├── src/
│   ├── config.py              # Configuration & constants
│   ├── fetch_data.py          # Data acquisition
│   ├── feature_pipeline.py    # Feature engineering
│   └── train_model.py         # Model training
├── app.py                      # Streamlit dashboard
├── requirements.txt            # Python dependencies
├── .env                        # Environment variables (not in Git)
├── .gitignore                 # Git ignore rules
└── README.md                  # This file
```

## 🤖 Automation (Optional)

To enable automated hourly data updates and daily model retraining:

1. **Push code to GitHub:**
   ```powershell
   git init
   git add .
   git commit -m "Initial commit"
   git remote add origin <your-repo-url>
   git push -u origin main
   ```

2. **Add GitHub Secrets:**
   - Go to your repository → Settings → Secrets and variables → Actions
   - Add secret: `HOPSWORKS_API_KEY` with your Hopsworks API key

3. **Enable GitHub Actions:**
   - Go to Actions tab and enable workflows

The pipeline will now:
- Fetch new data every hour
- Retrain models daily at 2 AM UTC
- Automatically update the Streamlit app with the best model

## 📈 Model Performance

The system trains and evaluates three model types:
- **Linear Regression**: Fast baseline model
- **Random Forest**: Handles non-linear relationships
- **XGBoost**: Typically the best performer

All models are saved daily, and the dashboard automatically uses the best one based on R² scores.

## 🌍 API Data Sources

- **Air Quality**: [Open-Meteo Air Quality API](https://open-meteo.com/en/docs/air-quality-api)
- **Weather**: [Open-Meteo Historical Weather API](https://open-meteo.com/en/docs/historical-weather-api)

## 📝 License

MIT License - feel free to use this project for learning and development!

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📧 Contact

For questions or suggestions, please open an issue on GitHub.
