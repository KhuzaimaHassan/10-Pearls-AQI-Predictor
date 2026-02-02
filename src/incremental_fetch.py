"""
Incremental Data Fetching for Hourly Updates
Fetches only the last few hours of new data and uploads to Hopsworks
"""
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import sys

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import (
    LATITUDE, LONGITUDE, LOCATION_NAME,
    OPEN_METEO_AIR_QUALITY_API, OPEN_METEO_WEATHER_API,
    AIR_QUALITY_PARAMS, WEATHER_PARAMS,
    TARGET_VARIABLE, LAG_HOURS, ROLLING_WINDOWS,
    HOPSWORKS_API_KEY, HOPSWORKS_PROJECT_NAME,
    FEATURE_GROUP_NAME, FEATURE_GROUP_VERSION
)

# Fetch last 6 hours (buffer to handle delays and missed runs)
FETCH_HOURS = 6

def fetch_incremental_air_quality():
    """
    Fetch recent air quality data (last few hours)
    """
    end_time = datetime.now()
    start_time = end_time - timedelta(hours=FETCH_HOURS)
    
    print(f"📡 Fetching incremental air quality data...")
    print(f"   Time range: {start_time.strftime('%Y-%m-%d %H:00')} to {end_time.strftime('%Y-%m-%d %H:00')}")
    
    params = {
        "latitude": LATITUDE,
        "longitude": LONGITUDE,
        "hourly": ",".join(AIR_QUALITY_PARAMS),
        "start_date": start_time.strftime("%Y-%m-%d"),
        "end_date": end_time.strftime("%Y-%m-%d"),
        "timezone": "Asia/Karachi"
    }
    
    try:
        response = requests.get(OPEN_METEO_AIR_QUALITY_API, params=params, timeout=30)
        response.raise_for_status()
        
        data = response.json()
        
        if "hourly" not in data:
            print("⚠️  Warning: No hourly data in response")
            return None
        
        df = pd.DataFrame(data["hourly"])
        df["time"] = pd.to_datetime(df["time"])
        
        print(f"✅ Fetched {len(df)} air quality records")
        return df
        
    except Exception as e:
        print(f"❌ Error fetching air quality: {e}")
        return None

def fetch_incremental_weather():
    """
    Fetch recent weather data using forecast API (has recent past data)
    """
    print(f"\n📡 Fetching incremental weather data...")
    
    # Use forecast API with past_hours for recent data
    params = {
        "latitude": LATITUDE,
        "longitude": LONGITUDE,
        "hourly": ",".join(WEATHER_PARAMS),
        "past_hours": FETCH_HOURS,
        "forecast_hours": 1,  # Minimal forecast
        "timezone": "Asia/Karachi"
    }
    
    try:
        response = requests.get(OPEN_METEO_WEATHER_API, params=params, timeout=30)
        response.raise_for_status()
        
        data = response.json()
        
        if "hourly" not in data:
            print("⚠️  Warning: No hourly data in response")
            return None
        
        df = pd.DataFrame(data["hourly"])
        df["time"] = pd.to_datetime(df["time"])
        
        # Filter to only past data (not forecast)
        df = df[df["time"] <= datetime.now()]
        
        print(f"✅ Fetched {len(df)} weather records")
        return df
        
    except Exception as e:
        print(f"❌ Error fetching weather: {e}")
        return None

def create_basic_features(df):
    """
    Create minimal features for incremental data
    Note: Lag features will be filled from historical data in Hopsworks
    """
    print("\n🔄 Creating incremental features...")
    
    # Cyclical time features
    df['hour_sin'] = np.sin(2 * np.pi * df["time"].dt.hour / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df["time"].dt.hour / 24)
    df['day_of_week_sin'] = np.sin(2 * np.pi * df["time"].dt.dayofweek / 7)
    df['day_of_week_cos'] = np.cos(2 * np.pi * df["time"].dt.dayofweek / 7)
    df['month_sin'] = np.sin(2 * np.pi * df["time"].dt.month / 12)
    df['month_cos'] = np.cos(2 * np.pi * df["time"].dt.month / 12)
    
    # Temporal features
    df['day_of_week_num'] = df["time"].dt.dayofweek
    df['is_weekend'] = df['day_of_week_num'].isin([5, 6]).astype(int)
    df['hour_of_day'] = df["time"].dt.hour
    
    # Rush hour
    rush_hours_morning = df['hour_of_day'].isin([7, 8, 9])
    rush_hours_evening = df['hour_of_day'].isin([17, 18, 19])
    df['is_rush_hour'] = (rush_hours_morning | rush_hours_evening).astype(int)
    
    # Season
    month = df["time"].dt.month
    df['season'] = 0
    df.loc[month.isin([12, 1, 2]), 'season'] = 0  # Winter
    df.loc[month.isin([3, 4, 5]), 'season'] = 1   # Spring
    df.loc[month.isin([6, 7, 8]), 'season'] = 2   # Summer
    df.loc[month.isin([9, 10, 11]), 'season'] = 3  # Fall
    
    # Wind components
    if 'wind_speed_10m' in df.columns and 'wind_direction_10m' in df.columns:
        wind_dir_rad = np.radians(df['wind_direction_10m'])
        df['wind_u_component'] = -df['wind_speed_10m'] * np.sin(wind_dir_rad)
        df['wind_v_component'] = -df['wind_speed_10m'] * np.cos(wind_dir_rad)
    
    # Interactions
    if 'temperature_2m' in df.columns and 'relative_humidity_2m' in df.columns:
        df['temp_humidity_interaction'] = df['temperature_2m'] * df['relative_humidity_2m']
    
    if 'wind_speed_10m' in df.columns and 'precipitation' in df.columns:
        df['wind_precip_interaction'] = df['wind_speed_10m'] * df['precipitation']
    
    # Placeholder lag and rolling features (will be NaN, Hopsworks handles this)
    for lag in LAG_HOURS:
        df[f"{TARGET_VARIABLE}_lag_{lag}h"] = np.nan
    
    for window in ROLLING_WINDOWS:
        df[f"{TARGET_VARIABLE}_rolling_mean_{window}h"] = np.nan
        if window >= 24:
            df[f"{TARGET_VARIABLE}_rolling_std_{window}h"] = np.nan
    
    print(f"✅ Created {len(df.columns)} feature columns")
    return df

def upload_incremental_to_hopsworks(df):
    """
    Upload incremental data to Hopsworks Feature Store
    Hopsworks will automatically deduplicate based on primary key (time)
    """
    print("\n☁️  Uploading incremental data to Hopsworks...")
    
    try:
        import hopsworks
        
        # Connect
        print("   Connecting to Hopsworks...")
        project = hopsworks.login(api_key_value=HOPSWORKS_API_KEY, project=HOPSWORKS_PROJECT_NAME)
        fs = project.get_feature_store()
        
        # Get existing feature group
        print(f"   Getting feature group: {FEATURE_GROUP_NAME}")
        fg = fs.get_feature_group(name=FEATURE_GROUP_NAME, version=FEATURE_GROUP_VERSION)
        
        # Insert new data (will deduplicate automatically)
        print(f"   Inserting {len(df)} records...")
        fg.insert(df, write_options={"wait_for_job": True})
        
        print("✅ Successfully uploaded incremental data!")
        print(f"   Records uploaded: {len(df)}")
        print(f"   Latest timestamp: {df['time'].max()}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error uploading to Hopsworks: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """
    Main function for incremental data fetching
    """
    print("=" * 60)
    print("🚀 Incremental Data Fetch - Hourly Update")
    print("=" * 60)
    print(f"Location: {LOCATION_NAME} ({LATITUDE}, {LONGITUDE})")
    print(f"Fetching last {FETCH_HOURS} hours of data")
    print("=" * 60)
    
    # Fetch air quality
    aq_df = fetch_incremental_air_quality()
    if aq_df is None:
        print("\n❌ Failed to fetch air quality data")
        return
    
    # Fetch weather
    weather_df = fetch_incremental_weather()
    if weather_df is None:
        print("\n❌ Failed to fetch weather data")
        return
    
    # Merge
    print("\n🔗 Merging data...")
    merged_df = pd.merge(aq_df, weather_df, on="time", how="inner")
    print(f"✅ Merged: {len(merged_df)} records")
    
    # Remove duplicates (in case API returns duplicates)
    initial_count = len(merged_df)
    merged_df = merged_df.drop_duplicates(subset=['time'], keep='last')
    merged_df = merged_df.sort_values('time').reset_index(drop=True)
    
    if len(merged_df) < initial_count:
        print(f"   Removed {initial_count - len(merged_df)} duplicate timestamps")
    
    # Basic validation
    if merged_df.empty:
        print("\n❌ No data to upload")
        return
    
    if TARGET_VARIABLE not in merged_df.columns or merged_df[TARGET_VARIABLE].isnull().all():
        print(f"\n❌ Target variable '{TARGET_VARIABLE}' is missing or all null")
        return
    
    # Create features
    feature_df = create_basic_features(merged_df)
    
    # Upload to Hopsworks
    success = upload_incremental_to_hopsworks(feature_df)
    
    print("\n" + "=" * 60)
    if success:
        print("✅ Incremental fetch completed successfully!")
        print(f"📊 New records added: {len(feature_df)}")
        print(f"📅 Time range: {feature_df['time'].min()} to {feature_df['time'].max()}")
    else:
        print("❌ Incremental fetch failed")
    print("=" * 60)

if __name__ == "__main__":
    main()
