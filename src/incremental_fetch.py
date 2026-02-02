"""
SMART Incremental Data Fetching for Hourly Updates
Queries Hopsworks first to find latest timestamp, then fetches only NEW data
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

# Fetch last 24 hours as fallback (if can't query Hopsworks)
FALLBACK_HOURS = 24

def get_latest_timestamp_from_hopsworks():
    """
    Query Hopsworks to find the latest timestamp we already have
    This prevents re-fetching and updating existing data
    """
    print("\n🔍 Checking latest timestamp in Hopsworks...")
    
    try:
        import hopsworks
        
        # Connect
        project = hopsworks.login(api_key_value=HOPSWORKS_API_KEY, project=HOPSWORKS_PROJECT_NAME)
        fs = project.get_feature_store()
        
        # Get feature group
        fg = fs.get_feature_group(name=FEATURE_GROUP_NAME, version=FEATURE_GROUP_VERSION)
        
        # Read entire dataset and find max time (simpler approach)
        print("   Reading feature group metadata...")
        df = fg.read()
        
        if not df.empty and 'time' in df.columns:
            latest_time = pd.to_datetime(df['time'].max())
            print(f"   ✅ Latest data in Hopsworks: {latest_time}")
            return latest_time
        else:
            print("   No data in Hopsworks yet")
            return None
            
    except Exception as e:
        print(f"   ⚠️  Could not check Hopsworks: {e}")
        print("   Will fetch last 24 hours as fallback")
        return None

def fetch_data_from_api(start_time, end_time):
    """
    Fetch air quality and weather data from Open-Meteo API
    """
    print(f"\n📡 Fetching data from Open-Meteo API...")
    print(f"   Range: {start_time.strftime('%Y-%m-%d %H:00')} to {end_time.strftime('%Y-%m-%d %H:00')}")
    
    # Air Quality
    aq_params = {
        "latitude": LATITUDE,
        "longitude": LONGITUDE,
        "hourly": ",".join(AIR_QUALITY_PARAMS),
        "start_date": start_time.strftime("%Y-%m-%d"),
        "end_date": end_time.strftime("%Y-%m-%d"),
        "timezone": "Asia/Karachi"
    }
    
    try:
        aq_response = requests.get(OPEN_METEO_AIR_QUALITY_API, params=aq_params, timeout=30)
        aq_response.raise_for_status()
        aq_data = aq_response.json()
        
        aq_df = pd.DataFrame(aq_data["hourly"])
        aq_df["time"] = pd.to_datetime(aq_df["time"])
        print(f"   ✅ Air quality: {len(aq_df)} records")
        
    except Exception as e:
        print(f"   ❌ Air quality fetch failed: {e}")
        return None
    
    # Weather (use forecast API for recent data)
    hours_diff = int((datetime.now() - start_time).total_seconds() / 3600) + 12
    weather_params = {
        "latitude": LATITUDE,
        "longitude": LONGITUDE,
        "hourly": ",".join(WEATHER_PARAMS),
        "past_hours": min(hours_diff, 168),  # Max 7 days
        "forecast_hours": 1,
        "timezone": "Asia/Karachi"
    }
    
    try:
        weather_response = requests.get(OPEN_METEO_WEATHER_API, params=weather_params, timeout=30)
        weather_response.raise_for_status()
        weather_data = weather_response.json()
        
        weather_df = pd.DataFrame(weather_data["hourly"])
        weather_df["time"] = pd.to_datetime(weather_df["time"])
        weather_df = weather_df[weather_df["time"] <= datetime.now()]  # Remove forecast
        print(f"   ✅ Weather: {len(weather_df)} records")
        
    except Exception as e:
        print(f"   ❌ Weather fetch failed: {e}")
        return None
    
    # Merge
    merged_df = pd.merge(aq_df, weather_df, on="time", how="inner")
    print(f"   ✅ Merged: {len(merged_df)} records")
    
    return merged_df

def filter_new_data(df, latest_in_hopsworks):
    """
    Filter to only rows with timestamps AFTER what's in Hopsworks
    """
    if latest_in_hopsworks is None:
        print("\n📊 No filtering needed (first run)")
        return df
    
    print(f"\n🔍 Filtering for NEW data only...")
    print(f"   Latest in Hopsworks: {latest_in_hopsworks}")
    
    initial_count = len(df)
    df_new = df[df['time'] > latest_in_hopsworks].copy()
    
    print(f"   Filtered: {initial_count} → {len(df_new)} NEW records")
    
    if not df_new.empty:
        print(f"   New data range: {df_new['time'].min()} to {df_new['time'].max()}")
    
    return df_new

def create_features(df):
    """Create all required features"""
    print("\n🔄 Creating features...")
    
    # Cyclical time
    df['hour_sin'] = np.sin(2 * np.pi * df["time"].dt.hour / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df["time"].dt.hour / 24)
    df['day_of_week_sin'] = np.sin(2 * np.pi * df["time"].dt.dayofweek / 7)
    df['day_of_week_cos'] = np.cos(2 * np.pi * df["time"].dt.dayofweek / 7)
    df['month_sin'] = np.sin(2 * np.pi * df["time"].dt.month / 12)
    df['month_cos'] = np.cos(2 * np.pi * df["time"].dt.month / 12)
    
    # Temporal
    df['day_of_week_num'] = df["time"].dt.dayofweek
    df['is_weekend'] = df['day_of_week_num'].isin([5, 6]).astype(int)
    df['hour_of_day'] = df["time"].dt.hour
    rush_hours_morning = df['hour_of_day'].isin([7, 8, 9])
    rush_hours_evening = df['hour_of_day'].isin([17, 18, 19])
    df['is_rush_hour'] = (rush_hours_morning | rush_hours_evening).astype(int)
    
    # Season
    month = df["time"].dt.month
    df['season'] = 0
    df.loc[month.isin([12, 1, 2]), 'season'] = 0
    df.loc[month.isin([3, 4, 5]), 'season'] = 1
    df.loc[month.isin([6, 7, 8]), 'season'] = 2
    df.loc[month.isin([9, 10, 11]), 'season'] = 3
    
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
    
    # Lag and rolling features (set to NaN)
    for lag in LAG_HOURS:
        df[f"{TARGET_VARIABLE}_lag_{lag}h"] = np.nan
    
    for window in ROLLING_WINDOWS:
        df[f"{TARGET_VARIABLE}_rolling_mean_{window}h"] = np.nan
        if window >= 24:
            df[f"{TARGET_VARIABLE}_rolling_std_{window}h"] = np.nan
    
    print(f"   ✅ Created {len(df.columns)} features")
    return df

def upload_to_hopsworks(df):
    """Upload to Hopsworks Feature Store"""
    print("\n☁️  Uploading to Hopsworks...")
    
    try:
        import hopsworks
        
        project = hopsworks.login(api_key_value=HOPSWORKS_API_KEY, project=HOPSWORKS_PROJECT_NAME)
        fs = project.get_feature_store()
        fg = fs.get_feature_group(name=FEATURE_GROUP_NAME, version=FEATURE_GROUP_VERSION)
        
        print(f"   Inserting {len(df)} records...")
        fg.insert(df, write_options={"wait_for_job": True})
        
        print(f"✅ Upload successful!")
        print(f"   Records: {len(df)}")
        print(f"   Latest: {df['time'].max()}")
        return True
        
    except Exception as e:
        print(f"❌ Upload failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("=" * 70)
    print("🚀 SMART Incremental Data Fetch")
    print("=" * 70)
    print(f"Location: {LOCATION_NAME}")
    print(f"Strategy: Query Hopsworks latest → Fetch only NEW data from API")
    print("=" * 70)
    
    # Step 1: Get latest timestamp from Hopsworks
    latest_in_hopsworks = get_latest_timestamp_from_hopsworks()
    
    # Step 2: Determine fetch range
    end_time = datetime.now()
    if latest_in_hopsworks:
        # Fetch from 1 hour before latest (buffer) to now
        start_time = latest_in_hopsworks - timedelta(hours=1)
        print(f"\n📅 Fetch range: {start_time} to {end_time}")
    else:
        # First run - fetch last 24 hours
        start_time = end_time - timedelta(hours=FALLBACK_HOURS)
        print(f"\n📅 First run - fetching last {FALLBACK_HOURS} hours")
    
    # Step 3: Fetch from API
    df = fetch_data_from_api(start_time, end_time)
    if df is None or df.empty:
        print("\n❌ No data fetched from API")
        return
    
    # Step 4: Filter to only NEW data
    df_new = filter_new_data(df, latest_in_hopsworks)
    
    if df_new.empty:
        print("\n⚠️  No NEW data to add (API has same old data)")
        print("   This means Open-Meteo API doesn't have newer data yet")
        print("   Will try again next hour")
        return
    
    # Step 5: Remove duplicates
    df_new = df_new.drop_duplicates(subset=['time'], keep='last')
    df_new = df_new.sort_values('time').reset_index(drop=True)
    
    # Step 6: Create features
    df_features = create_features(df_new)
    
    # Step 7: Upload
    success = upload_to_hopsworks(df_features)
    
    print("\n" + "=" * 70)
    if success:
        print("✅ SMART INCREMENTAL FETCH COMPLETE!")
        print(f"📊 Added {len(df_features)} NEW records to Hopsworks")
        print(f"📅 Range: {df_features['time'].min()} to {df_features['time'].max()}")
    else:
        print("❌ Fetch failed")
    print("=" * 70)

if __name__ == "__main__":
    main()
