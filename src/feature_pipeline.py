"""
Feature Engineering Pipeline for Pearls AQI Predictor
Creates lag features, rolling statistics, cyclical features, and uploads to Hopsworks
"""
import pandas as pd
import numpy as np
import os
import sys
from datetime import datetime

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import (
    RAW_DATA_FILE, PROCESSED_DATA_FILE, PROCESSED_DATA_DIR,
    TARGET_VARIABLE, LAG_HOURS, ROLLING_WINDOWS,
    HOPSWORKS_API_KEY, HOPSWORKS_PROJECT_NAME,
    FEATURE_GROUP_NAME, FEATURE_GROUP_VERSION
)

def load_raw_data():
    """
    Load raw data from CSV
    
    Returns:
        pandas.DataFrame: Raw data
    """
    print("=" * 60)
    print("📂 Loading raw data...")
    print("=" * 60)
    
    if not os.path.exists(RAW_DATA_FILE):
        print(f"❌ Error: Raw data file not found at {RAW_DATA_FILE}")
        print("   Please run src/fetch_data.py first")
        return None
    
    df = pd.read_csv(RAW_DATA_FILE)
    df['time'] = pd.to_datetime(df['time'])
    
    initial_count = len(df)
    print(f"✅ Loaded {initial_count} records")
    print(f"   Date range: {df['time'].min()} to {df['time'].max()}")
    print(f"   Columns: {list(df.columns)}")
    
    # === DUPLICATE DETECTION & REMOVAL ===
    # Critical: Remove duplicate timestamps to prevent bias in hourly fetches
    # If API returns same data multiple times, we only keep the first occurrence
    print("\n🔍 Checking for duplicate timestamps...")
    
    # Check for exact duplicates (same time + same values)
    duplicates_exact = df.duplicated(keep='first').sum()
    if duplicates_exact > 0:
        df = df.drop_duplicates(keep='first')
        print(f"   Removed {duplicates_exact} exact duplicate records")
    
    # Check for duplicate timestamps (same time, different values - keep latest)
    duplicates_time = df.duplicated(subset=['time'], keep='last').sum()
    if duplicates_time > 0:
        df = df.drop_duplicates(subset=['time'], keep='last')
        print(f"   Removed {duplicates_time} duplicate timestamps (kept latest values)")
    
    # Sort by time to ensure chronological order
    df = df.sort_values('time').reset_index(drop=True)
    
    final_count = len(df)
    removed = initial_count - final_count
    
    if removed > 0:
        print(f"✅ Duplicate handling complete: {removed} records removed")
        print(f"   Final dataset: {final_count} unique hourly records")
    else:
        print(f"✅ No duplicates found - data quality good!")
    
    return df

def create_lag_features(df):
    """
    Create lag features for the target variable
    
    Args:
        df (pandas.DataFrame): Input dataframe
        
    Returns:
        pandas.DataFrame: Dataframe with lag features
    """
    print("\n🔄 Creating lag features...")
    
    for lag in LAG_HOURS:
        col_name = f"{TARGET_VARIABLE}_lag_{lag}h"
        df[col_name] = df[TARGET_VARIABLE].shift(lag)
        print(f"   Created: {col_name}")
    
    return df

def create_rolling_features(df):
    """
    Create rolling statistics features
    
    Args:
        df (pandas.DataFrame): Input dataframe
        
    Returns:
        pandas.DataFrame: Dataframe with rolling features
    """
    print("\n📊 Creating rolling statistics features...")
    
    for window in ROLLING_WINDOWS:
        # Rolling mean
        col_name = f"{TARGET_VARIABLE}_rolling_mean_{window}h"
        df[col_name] = df[TARGET_VARIABLE].rolling(window=window, min_periods=1).mean()
        print(f"   Created: {col_name}")
        
        # Rolling std (volatility)
        if window >= 24:  # Only for larger windows
            col_name = f"{TARGET_VARIABLE}_rolling_std_{window}h"
            df[col_name] = df[TARGET_VARIABLE].rolling(window=window, min_periods=1).std()
            print(f"   Created: {col_name}")
    
    return df

def create_cyclical_features(df):
    """
    Create cyclical date features using sine/cosine transformation
    
    Args:
        df (pandas.DataFrame): Input dataframe
        
    Returns:
        pandas.DataFrame: Dataframe with cyclical features
    """
    print("\n🌀 Creating cyclical date features...")
    
    # Extract time components
    df['hour'] = df['time'].dt.hour
    df['day_of_week'] = df['time'].dt.dayofweek
    df['month'] = df['time'].dt.month
    
    # Hour (0-23)
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    print("   Created: hour_sin, hour_cos")
    
    # Day of week (0-6)
    df['day_of_week_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['day_of_week_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
    print("   Created: day_of_week_sin, day_of_week_cos")
    
    # Month (1-12)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    print("   Created: month_sin, month_cos")
    
    # Drop intermediate columns
    df = df.drop(['hour', 'day_of_week', 'month'], axis=1)
    
    return df

def create_wind_components(df):
    """
    Create wind U and V components from speed and direction
    Wind components are critical for AQI prediction as pollutants drift with wind
    
    U-component: East-West wind (positive = eastward)
    V-component: North-South wind (positive = northward)
    
    Args:
        df (pandas.DataFrame): Input dataframe
        
    Returns:
        pandas.DataFrame: Dataframe with wind components
    """
    print("\n💨 Creating wind U/V components...")
    
    if 'wind_speed_10m' in df.columns and 'wind_direction_10m' in df.columns:
        # Convert wind direction from degrees to radians
        # Meteorological convention: 0° = North, 90° = East, 180° = South, 270° = West
        # Convert to math convention for U/V calculation
        wind_dir_rad = np.radians(df['wind_direction_10m'])
        
        # Calculate U (east-west) and V (north-south) components
        # U = -speed * sin(direction) [negative because meteorological convention]
        # V = -speed * cos(direction)
        df['wind_u_component'] = -df['wind_speed_10m'] * np.sin(wind_dir_rad)
        df['wind_v_component'] = -df['wind_speed_10m'] * np.cos(wind_dir_rad)
        
        print("   Created: wind_u_component (east-west)")
        print("   Created: wind_v_component (north-south)")
        
        # Wind magnitude check (should equal wind_speed_10m)
        # This is a sanity check
        wind_magnitude = np.sqrt(df['wind_u_component']**2 + df['wind_v_component']**2)
        diff = np.abs(wind_magnitude - df['wind_speed_10m']).mean()
        print(f"   ✓ Component validation: avg difference = {diff:.6f} (should be ~0)")
    else:
        print("   ⚠️ Wind speed or direction not found, skipping wind components")
    
    return df

def create_temporal_features(df):
    """
    Create additional temporal features beyond cyclical encoding
    These capture weekly patterns, rush hours, etc.
    
    Args:
        df (pandas.DataFrame): Input dataframe
        
    Returns:
        pandas.DataFrame: Dataframe with temporal features
    """
    print("\n📅 Creating additional temporal features...")
    
    # Day of week as categorical (0 = Monday, 6 = Sunday)
    df['day_of_week_num'] = df['time'].dt.dayofweek
    
    # Weekend indicator (Friday-Saturday in Pakistan, but using Saturday-Sunday for international standard)
    df['is_weekend'] = df['day_of_week_num'].isin([5, 6]).astype(int)
    print("   Created: day_of_week_num, is_weekend")
    
    # Hour of day (0-23) - keep as numeric for some models
    df['hour_of_day'] = df['time'].dt.hour
    
    # Rush hour indicator (7-9 AM and 5-7 PM when traffic pollution peaks)
    rush_hours_morning = df['hour_of_day'].isin([7, 8, 9])
    rush_hours_evening = df['hour_of_day'].isin([17, 18, 19])
    df['is_rush_hour'] = (rush_hours_morning | rush_hours_evening).astype(int)
    print("   Created: hour_of_day, is_rush_hour")
    
    # Season (for easier interpretation, though cyclical month captures this)
    # Winter: Dec-Feb (12, 1, 2), Spring: Mar-May (3,4,5), Summer: Jun-Aug (6,7,8), Fall: Sep-Nov (9,10,11)
    month = df['time'].dt.month
    df['season'] = 0  # Default
    df.loc[month.isin([12, 1, 2]), 'season'] = 0  # Winter
    df.loc[month.isin([3, 4, 5]), 'season'] = 1   # Spring  
    df.loc[month.isin([6, 7, 8]), 'season'] = 2   # Summer
    df.loc[month.isin([9, 10, 11]), 'season'] = 3  # Fall
    print("   Created: season (0=Winter, 1=Spring, 2=Summer, 3=Fall)")
    
    return df

def create_interaction_features(df):
    """
    Create interaction features between weather variables
    
    Args:
        df (pandas.DataFrame): Input dataframe
        
    Returns:
        pandas.DataFrame: Dataframe with interaction features
    """
    print("\n🔗 Creating interaction features...")
    
    # Temperature × Humidity
    if 'temperature_2m' in df.columns and 'relative_humidity_2m' in df.columns:
        df['temp_humidity_interaction'] = df['temperature_2m'] * df['relative_humidity_2m']
        print("   Created: temp_humidity_interaction")
    
    # Wind speed × Precipitation (weather severity)
    if 'wind_speed_10m' in df.columns and 'precipitation' in df.columns:
        df['wind_precip_interaction'] = df['wind_speed_10m'] * df['precipitation']
        print("   Created: wind_precip_interaction")
    
    return df

def handle_missing_values(df):
    """
    Handle missing values in the dataset
    
    Args:
        df (pandas.DataFrame): Input dataframe
        
    Returns:
        pandas.DataFrame: Dataframe with handled missing values
    """
    print("\n🔧 Handling missing values...")
    
    # Check initial missing values
    missing_before = df.isnull().sum().sum()
    print(f"   Missing values before: {missing_before}")
    
    # Drop rows where target variable is missing
    initial_len = len(df)
    df = df.dropna(subset=[TARGET_VARIABLE])
    dropped = initial_len - len(df)
    if dropped > 0:
        print(f"   Dropped {dropped} rows with missing target variable")
    
    # For lag features, dropping rows is acceptable since we're creating history-based features
    # Drop rows where any lag feature is NaN (typically the first few rows)
    lag_cols = [col for col in df.columns if 'lag' in col]
    if lag_cols:
        initial_len = len(df)
        df = df.dropna(subset=lag_cols)
        dropped = initial_len - len(df)
        if dropped > 0:
            print(f"   Dropped {dropped} rows with missing lag features (expected for first {max(LAG_HOURS)} hours)")
    
    # Fill remaining missing values with forward fill, then backward fill
    df = df.fillna(method='ffill').fillna(method='bfill')
    
    missing_after = df.isnull().sum().sum()
    print(f"   Missing values after: {missing_after}")
    
    if missing_after > 0:
        print(f"   ⚠️  Warning: {missing_after} missing values remain")
        # Drop any remaining rows with missing values
        df = df.dropna()
        print(f"   Dropped remaining rows with missing values. Final count: {len(df)}")
    
    return df

def upload_to_hopsworks(df):
    """
    Upload features to Hopsworks Feature Store
    
    Args:
        df (pandas.DataFrame): Features dataframe
    """
    print("\n☁️  Uploading to Hopsworks Feature Store...")
    
    if not HOPSWORKS_API_KEY or HOPSWORKS_API_KEY == "your_api_key_here":
        print("⚠️  Skipping Hopsworks upload: API key not configured")
        print("  To upload to Hopsworks:")
        print("  1. Create account at https://www.hopsworks.ai/")
        print("  2. Create a project")
        print("  3. Get API key from Settings → API Keys")
        print("  4. Add to .env file: HOPSWORKS_API_KEY=<your_key>")
        return False
    
    try:
        import hopsworks
        
        # Connect to Hopsworks
        print("   Connecting to Hopsworks...")
        project = hopsworks.login(api_key_value=HOPSWORKS_API_KEY, project=HOPSWORKS_PROJECT_NAME)
        fs = project.get_feature_store()
        
        # Create or get feature group
        print(f"   Creating/updating feature group: {FEATURE_GROUP_NAME}")
        
        # Prepare dataframe for Hopsworks
        # The 'time' column will be the event time
        feature_df = df.copy()
        
        # Create feature group
        fg = fs.get_or_create_feature_group(
            name=FEATURE_GROUP_NAME,
            version=FEATURE_GROUP_VERSION,
            description="Karachi AQI features with lag, rolling stats, and cyclical features",
            primary_key=["time"],
            event_time="time",
            online_enabled=False
        )
        
        # Insert data
        print("   Uploading data...")
        fg.insert(feature_df, write_options={"wait_for_job": True})
        
        print("✅ Successfully uploaded to Hopsworks!")
        print(f"   Feature Group: {FEATURE_GROUP_NAME} (version {FEATURE_GROUP_VERSION})")
        print(f"   Records uploaded: {len(feature_df)}")
        
        return True
        
    except ImportError as e:
        print(f"❌ Error importing hopsworks: {e}")
        print("  Install with: pip install hopsworks")
        import traceback
        traceback.print_exc()
        return False
    except Exception as e:
        print(f"❌ Error uploading to Hopsworks: {e}")
        import traceback
        traceback.print_exc()
        return False

def save_features_locally(df):
    """
    Save engineered features to local CSV
    
    Args:
        df (pandas.DataFrame): Features dataframe
    """
    print("\n💾 Saving features locally...")
    
    # Create directory if it doesn't exist
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
    
    # Save to CSV
    df.to_csv(PROCESSED_DATA_FILE, index=False)
    
    file_size = os.path.getsize(PROCESSED_DATA_FILE) / (1024 * 1024)  # MB
    print(f"✅ Features saved to {PROCESSED_DATA_FILE}")
    print(f"   File size: {file_size:.2f} MB")
    print(f"   Records: {len(df)}")
    print(f"   Features: {len(df.columns)}")

def main():
    """
    Main function to orchestrate feature engineering
    """
    print("\n" + "=" * 60)
    print("🚀 Pearls AQI Predictor - Feature Engineering")
    print("=" * 60)
    
    # Load raw data
    df = load_raw_data()
    if df is None:
        return
    
    # Create lag features
    df = create_lag_features(df)
    
    # Create rolling statistics
    df = create_rolling_features(df)
    
    # Create wind U/V components (CRITICAL for AQI prediction)
    df = create_wind_components(df)
    
    # Create cyclical features
    df = create_cyclical_features(df)
    
    # Create additional temporal features (day of week, weekend, rush hour)
    df = create_temporal_features(df)
    
    # Create interaction features
    df = create_interaction_features(df)
    
    # Handle missing values
    df = handle_missing_values(df)
    
    # Display feature summary
    print("\n📋 Feature Engineering Summary:")
    print("=" * 60)
    print(f"Total features: {len(df.columns)}")
    print(f"Total records: {len(df)}")
    print(f"Date range: {df['time'].min()} to {df['time'].max()}")
    print("\nFeature columns:")
    for i, col in enumerate(df.columns, 1):
        print(f"  {i}. {col}")
    
    # Save features locally
    save_features_locally(df)
    
    # Upload to Hopsworks
    upload_success = upload_to_hopsworks(df)
    
    print("\n" + "=" * 60)
    if upload_success:
        print("✅ Feature engineering completed successfully!")
        print("   ☁️  Features uploaded to Hopsworks")
    else:
        print("✅ Feature engineering completed!")
        print("   📁 Features saved locally (Hopsworks upload skipped)")
    print("=" * 60)
    print(f"📁 Processed data: {PROCESSED_DATA_FILE}")
    print("=" * 60)

if __name__ == "__main__":
    main()
