"""
Configuration file for Pearls AQI Predictor
Contains constants, API endpoints, and configuration settings
"""
import os
from datetime import datetime, timedelta
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# ===== LOCATION SETTINGS =====
# Karachi, Pakistan coordinates
LATITUDE = 24.8607
LONGITUDE = 67.0011
LOCATION_NAME = "Karachi"

# ===== API ENDPOINTS =====
OPEN_METEO_AIR_QUALITY_API = "https://air-quality-api.open-meteo.com/v1/air-quality"
OPEN_METEO_WEATHER_API = "https://api.open-meteo.com/v1/forecast"
OPEN_METEO_ARCHIVE_API = "https://archive-api.open-meteo.com/v1/archive"

# ===== DATA CONFIGURATION =====
# Historical data period (in days from today)
# 1095 days = 3 years - captures full seasonal cycles and multi-year patterns
# This significantly reduces distribution shift and improves model generalization
HISTORICAL_DAYS = 1095  # Fetch 3 years of data (changed from 365)

# Calculate date range
END_DATE = datetime.now().date()
START_DATE = END_DATE - timedelta(days=HISTORICAL_DAYS)

# ===== FEATURE CONFIGURATION =====
# Air quality parameters to fetch
AIR_QUALITY_PARAMS = [
    "pm10",           # Particulate Matter < 10 μm
    "pm2_5",          # Particulate Matter < 2.5 μm
    "us_aqi",         # US Air Quality Index
    "european_aqi"    # European Air Quality Index
]

# Weather parameters to fetch
WEATHER_PARAMS = [
    "temperature_2m",      # Temperature at 2 meters
    "relative_humidity_2m", # Relative humidity
    "wind_speed_10m",      # Wind speed at 10 meters
    "wind_direction_10m",  # Wind direction (needed for U/V components)
    "precipitation"        # Precipitation
]

# Lag features to create (in hours)
LAG_HOURS = [1, 2, 3, 24]  # 1h, 2h, 3h, 24h (yesterday) - BASELINE THAT WORKED

# Rolling window sizes (in hours)
ROLLING_WINDOWS = [3, 24]  # 3h, 24h - BASELINE THAT WORKED

# Target variable for prediction
TARGET_VARIABLE = "us_aqi"  # Using US AQI as the main target

# ===== HOPSWORKS CONFIGURATION =====
HOPSWORKS_API_KEY = os.getenv("HOPSWORKS_API_KEY")
HOPSWORKS_PROJECT_NAME = os.getenv("HOPSWORKS_PROJECT_NAME", "pearls_aqi_predictor")
FEATURE_GROUP_NAME = "air_quality_karachi"
FEATURE_GROUP_VERSION = 1

# ===== MODEL CONFIGURATION =====
# Train/test split ratio
TRAIN_TEST_SPLIT = 0.5  # First 50% train, last 50% test

# Prediction horizons (in hours)
PREDICTION_HORIZONS = [24, 48, 72]  # Day 1, 2, 3

# Model types to train
MODEL_TYPES = ["linear_regression", "random_forest", "xgboost"]

# Model parameters
RANDOM_FOREST_PARAMS = {
    "n_estimators": 100,
    "max_depth": 20,
    "min_samples_split": 5,
    "random_state": 42
}

XGBOOST_PARAMS = {
    "n_estimators": 100,
    "max_depth": 6,
    "learning_rate": 0.1,
    "random_state": 42
}

# ===== FILE PATHS =====
RAW_DATA_DIR = "data/raw"
PROCESSED_DATA_DIR = "data/processed"
MODELS_DIR = "models"

RAW_DATA_FILE = f"{RAW_DATA_DIR}/historical_data.csv"
PROCESSED_DATA_FILE = f"{PROCESSED_DATA_DIR}/features.csv"

# ===== AQI THRESHOLDS =====
# US AQI categories
AQI_CATEGORIES = {
    "good": (0, 50, "#00E400"),           # Green
    "moderate": (51, 100, "#FFFF00"),      # Yellow
    "unhealthy_sensitive": (101, 150, "#FF7E00"),  # Orange
    "unhealthy": (151, 200, "#FF0000"),    # Red
    "very_unhealthy": (201, 300, "#8F3F97"),  # Purple
    "hazardous": (301, 500, "#7E0023")     # Maroon
}

def get_aqi_category(aqi_value):
    """
    Get AQI category and color based on AQI value
    
    Args:
        aqi_value (float): AQI value
        
    Returns:
        tuple: (category_name, color)
    """
    for category, (min_val, max_val, color) in AQI_CATEGORIES.items():
        if min_val <= aqi_value <= max_val:
            return category.replace("_", " ").title(), color
    
    # If AQI is above 500
    if aqi_value > 500:
        return "Hazardous", "#7E0023"
    
    # If AQI is below 0 (shouldn't happen, but handle it)
    return "Unknown", "#808080"

# ===== LOGGING =====
LOG_LEVEL = "INFO"

def get_aqi_color(aqi_value):
    """
    Get color for AQI value
    
    Args:
        aqi_value (float): AQI value
        
    Returns:
        str: Color hex code
    """
    for category, (min_val, max_val, color) in AQI_CATEGORIES.items():
        if min_val <= aqi_value <= max_val:
            return color
    
    # If AQI is above 500
    if aqi_value > 500:
        return '#7E0023'  # Hazardous
    
    # Default
    return '#808080'  # Gray for unknown
