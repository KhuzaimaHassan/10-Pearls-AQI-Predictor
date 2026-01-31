"""
Data Fetching Script for Pearls AQI Predictor
Fetches historical air quality and weather data from Open-Meteo API
"""
import requests
import pandas as pd
import json
from datetime import datetime
import os
import sys

# Add parent directory to path to import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import (
    LATITUDE, LONGITUDE, LOCATION_NAME,
    OPEN_METEO_AIR_QUALITY_API, OPEN_METEO_WEATHER_API,
    START_DATE, END_DATE,
    AIR_QUALITY_PARAMS, WEATHER_PARAMS,
    RAW_DATA_DIR, RAW_DATA_FILE
)

def fetch_air_quality_data():
    """
    Fetch historical air quality data from Open-Meteo API
    
    Returns:
        pandas.DataFrame: Air quality data with timestamp and AQI metrics
    """
    print(f"📡 Fetching air quality data for {LOCATION_NAME}...")
    print(f"   Date range: {START_DATE} to {END_DATE}")
    
    # Build API parameters
    params = {
        "latitude": LATITUDE,
        "longitude": LONGITUDE,
        "hourly": ",".join(AIR_QUALITY_PARAMS),
        "start_date": START_DATE.strftime("%Y-%m-%d"),
        "end_date": END_DATE.strftime("%Y-%m-%d"),
        "timezone": "Asia/Karachi"
    }
    
    try:
        # Make API request
        response = requests.get(OPEN_METEO_AIR_QUALITY_API, params=params, timeout=30)
        response.raise_for_status()
        
        data = response.json()
        
        # Check if we got valid data
        if "hourly" not in data:
            print("⚠️  Warning: No hourly data in air quality response!")
            print(f"   Response: {json.dumps(data, indent=2)[:500]}")
            return None
        
        # Convert to DataFrame
        df = pd.DataFrame(data["hourly"])
        df["time"] = pd.to_datetime(df["time"])
        
        print(f"✅ Fetched {len(df)} air quality records")
        print(f"   Columns: {', '.join(df.columns.tolist())}")
        
        return df
        
    except requests.exceptions.RequestException as e:
        print(f"❌ Error fetching air quality data: {e}")
        return None

def fetch_weather_data():
    """
    Fetch historical weather data from Open-Meteo API
    
    Returns:
        pandas.DataFrame: Weather data with timestamp and weather metrics
    """
    print(f"\n📡 Fetching weather data for {LOCATION_NAME}...")
    print(f"   Date range: {START_DATE} to {END_DATE}")
    
    # Open-Meteo forecast API only provides ~7 days of past data with past_days parameter
    # For longer history, we need to use the archive endpoint
    # But archive has a 5-day delay, so we'll combine both
    
    # For historical data older than 5 days, use archive API
    params = {
        "latitude": LATITUDE,
        "longitude": LONGITUDE,
        "hourly": ",".join(WEATHER_PARAMS),
        "start_date": START_DATE.strftime("%Y-%m-%d"),
        "end_date": END_DATE.strftime("%Y-%m-%d"),
        "timezone": "Asia/Karachi"
    }
    
    try:
        # Try archive API first
        archive_url = "https://archive-api.open-meteo.com/v1/archive"
        response = requests.get(archive_url, params=params, timeout=30)
        response.raise_for_status()
        
        data = response.json()
        
        # Check if we got valid data
        if "hourly" not in data:
            print("⚠️  Warning: No hourly data in weather response!")
            print(f"   Response: {json.dumps(data, indent=2)[:500]}")
            return None
        
        # Convert to DataFrame
        df = pd.DataFrame(data["hourly"])
        df["time"] = pd.to_datetime(df["time"])
        
        print(f"✅ Fetched {len(df)} weather records")
        print(f"   Columns: {', '.join(df.columns.tolist())}")
        
        return df
        
    except requests.exceptions.RequestException as e:
        print(f"❌ Error fetching weather data: {e}")
        return None

def merge_data(air_quality_df, weather_df):
    """
    Merge air quality and weather data on timestamp
    
    Args:
        air_quality_df (pandas.DataFrame): Air quality data
        weather_df (pandas.DataFrame): Weather data
        
    Returns:
        pandas.DataFrame: Merged data
    """
    print("\n🔗 Merging air quality and weather data...")
    
    if air_quality_df is None or weather_df is None:
        print("❌ Cannot merge: one or both DataFrames are None")
        return None
    
    # Merge on time column
    merged_df = pd.merge(air_quality_df, weather_df, on="time", how="inner")
    
    print(f"✅ Merged dataset: {len(merged_df)} records")
    print(f"   Total columns: {len(merged_df.columns)}")
    
    # Check for missing values
    missing_counts = merged_df.isnull().sum()
    missing_pct = (missing_counts / len(merged_df) * 100).round(2)
    
    print("\n📊 Missing values summary:")
    for col, pct in missing_pct.items():
        if pct > 0:
            print(f"   {col}: {pct}%")
    
    return merged_df

def validate_data(df):
    """
    Validate the fetched data
    
    Args:
        df (pandas.DataFrame): Data to validate
        
    Returns:
        bool: True if data is valid, False otherwise
    """
    print("\n🔍 Validating data...")
    
    if df is None or df.empty:
        print("❌ Validation failed: DataFrame is None or empty")
        return False
    
    # Check minimum number of records (at least 30 days of hourly data)
    min_records = 30 * 24  # 720 records
    if len(df) < min_records:
        print(f"⚠️  Warning: Only {len(df)} records (expected at least {min_records})")
    
    # Check for required columns
    required_cols = ["time", "us_aqi"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        print(f"❌ Missing required columns: {missing_cols}")
        return False
    
    # Check if target variable has sufficient non-null values
    us_aqi_valid = df["us_aqi"].notna().sum()
    us_aqi_pct = (us_aqi_valid / len(df) * 100)
    
    print(f"   ✅ Target variable (us_aqi): {us_aqi_pct:.1f}% valid values")
    
    if us_aqi_pct < 50:
        print(f"❌ Insufficient target data: only {us_aqi_pct:.1f}% valid")
        return False
    
    # Check data types
    print(f"   ✅ Data types: {df.dtypes.value_counts().to_dict()}")
    
    # Check date range
    print(f"   ✅ Date range: {df['time'].min()} to {df['time'].max()}")
    
    print("\n✅ Data validation passed!")
    return True

def save_data(df, filepath):
    """
    Save data to CSV file
    
    Args:
        df (pandas.DataFrame): Data to save
        filepath (str): Path to save the file
    """
    print(f"\n💾 Saving data to {filepath}...")
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    # Save to CSV
    df.to_csv(filepath, index=False)
    
    file_size = os.path.getsize(filepath) / (1024 * 1024)  # Convert to MB
    print(f"✅ Data saved successfully ({file_size:.2f} MB)")

def main():
    """
    Main function to orchestrate data fetching
    """
    print("=" * 60)
    print("🚀 Pearls AQI Predictor - Data Fetching")
    print("=" * 60)
    print(f"Location: {LOCATION_NAME} ({LATITUDE}, {LONGITUDE})")
    print(f"Date Range: {START_DATE} to {END_DATE}")
    print("=" * 60)
    
    # Fetch air quality data
    air_quality_df = fetch_air_quality_data()
    
    # Fetch weather data
    weather_df = fetch_weather_data()
    
    # Merge data
    merged_df = merge_data(air_quality_df, weather_df)
    
    # Validate data
    if not validate_data(merged_df):
        print("\n❌ Data validation failed. Please check the API responses.")
        return
    
    # Save data
    save_data(merged_df, RAW_DATA_FILE)
    
    print("\n" + "=" * 60)
    print("✅ Data fetching completed successfully!")
    print("=" * 60)
    print(f"📁 Raw data saved to: {RAW_DATA_FILE}")
    print(f"📊 Total records: {len(merged_df)}")
    print(f"📅 Coverage: {len(merged_df) / 24:.1f} days")
    print("=" * 60)

if __name__ == "__main__":
    main()
