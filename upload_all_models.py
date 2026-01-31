"""
Upload ALL 15 models to Hopsworks - Fixed version
"""
import os
import glob
import joblib
from dotenv import load_dotenv

load_dotenv()

HOPSWORKS_API_KEY = os.getenv('HOPSWORKS_API_KEY')
HOPSWORKS_PROJECT_NAME = os.getenv('HOPSWORKS_PROJECT_NAME', 'three_days_AQI')

print("="*80)
print("UPLOADING ALL 15 MODELS TO HOPSWORKS")
print("="*80)

try:
    import hopsworks
    
    print(f"\nConnecting to Hopsworks project: {HOPSWORKS_PROJECT_NAME}...")
    project = hopsworks.login(
        api_key_value=HOPSWORKS_API_KEY,
        project=HOPSWORKS_PROJECT_NAME
    )
    
    mr = project.get_model_registry()
    print(f"Connected successfully!\n")
    
    total_uploaded = 0
    
    for day in [1, 2, 3]:
        horizon_hours = day * 24
        
        print(f"\n{'='*80}")
        print(f"DAY {day} ({horizon_hours}h ahead)")
        print('='*80)
        
        # Get all models for this day  
        day_models = glob.glob(f'models/*_day{day}_*.pkl')
        
        if not day_models:
            print(f"  No models found for Day {day}")
            continue
        
        print(f"\nFound {len(day_models)} models for Day {day}")
        
        # Upload each model
        for model_path in day_models:
            try:
                # Determine model type from filename
                model_name = None
                if 'catboost' in model_path.lower():
                    model_name = 'catboost'
                elif 'ensemble' in model_path.lower():
                    model_name = 'ensemble'
                elif 'xgboost' in model_path.lower():
                    model_name = 'xgboost'
                elif 'lightgbm' in model_path.lower():
                    model_name = 'lightgbm'
                elif 'random' in model_path.lower():
                    model_name = 'random_forest'
                
                if not model_name:
                    continue
                
                # Load model data
                model_data = joblib.load(model_path)
                metrics = model_data['metrics']
                
                test_r2 = metrics.get('test_r2', 0)
                test_rmse = metrics.get('test_rmse', 0)
                test_mae = metrics.get('test_mae', 0)
                cv_r2 = metrics.get('cv_mean_test_r2', 0)
                
                print(f"\n  Uploading {model_name.upper()}:")
                print(f"    Test R2: {test_r2:.4f}, RMSE: {test_rmse:.2f}")
                
                # Create model in registry
                registry_name = f"pearls_aqi_day{day}_{model_name}"
                
                aqi_model = mr.python.create_model(
                    name=registry_name,
                    metrics={
                        "test_r2": float(test_r2),
                        "test_rmse": float(test_rmse),
                        "test_mae": float(test_mae),
                        "cv_mean_r2": float(cv_r2),
                        "horizon_hours": horizon_hours
                    },
                    description=f"{model_name.upper()} model for Day {day} ({horizon_hours}h) AQI prediction. "
                                f"Trained on 26,280 samples with 32 baseline features. "
                                f"Performance: R2={test_r2:.4f}, RMSE={test_rmse:.2f}"
                )
                
                # Save model file
                aqi_model.save(model_path)
                
                print(f"    SUCCESS: {registry_name}")
                total_uploaded += 1
                
            except Exception as e:
                print(f"    ERROR: {os.path.basename(model_path)}: {e}")
    
    print("\n" + "="*80)
    print(f"SUCCESS! UPLOADED {total_uploaded} MODELS TO HOPSWORKS")
    print("="*80)
    print(f"\nView at: https://c.app.hopsworks.ai/p/{project.id}/models")
    print("="*80)
    
except Exception as e:
    print(f"\nERROR: {e}")
    import traceback
    traceback.print_exc()
