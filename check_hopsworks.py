"""
Verify Hopsworks Uploads - Check what's currently in Hopsworks
"""
import os
from dotenv import load_dotenv

load_dotenv()

HOPSWORKS_API_KEY = os.getenv('HOPSWORKS_API_KEY')
HOPSWORKS_PROJECT_NAME = os.getenv('HOPSWORKS_PROJECT_NAME', 'three_days_AQI')

print("="*80)
print("🔍 CHECKING HOPSWORKS UPLOADS")
print("="*80)

if not HOPSWORKS_API_KEY or HOPSWORKS_API_KEY == 'your_api_key_here':
    print("\n❌ HOPSWORKS_API_KEY not configured!")
    print("Cannot check uploads without API key.")
    exit(1)

try:
    import hopsworks
    
    print(f"\n🔗 Connecting to Hopsworks project: {HOPSWORKS_PROJECT_NAME}...")
    project = hopsworks.login(
        api_key_value=HOPSWORKS_API_KEY,
        project=HOPSWORKS_PROJECT_NAME
    )
    print(f"✅ Connected successfully!\n")
    
    # Check Feature Store
    print("="*80)
    print("📊 FEATURE STORE STATUS")
    print("="*80)
    
    try:
        fs = project.get_feature_store()
        
        # Try to get feature group
        try:
            fg = fs.get_feature_group(name="aqi_features", version=1)
            
            # Get metadata
            print(f"\n✅ Feature Group Found: aqi_features (v1)")
            print(f"   Description: {fg.description}")
            
            # Try to read some data
            df = fg.read()
            print(f"\n📈 Feature Data:")
            print(f"   Total Records: {len(df):,}")
            print(f"   Features: {len(df.columns)}")
            print(f"   Date Range: {df['time'].min()} to {df['time'].max()}")
            print(f"\n   Features: {', '.join(df.columns[:10].tolist())}...")
            
            print("\n✅ FEATURES UPLOADED SUCCESSFULLY!")
            
        except Exception as e:
            print(f"\n⚠️  Feature Group 'aqi_features' not found")
            print(f"   Error: {e}")
            print("\n❌ FEATURES NOT UPLOADED YET")
            
    except Exception as e:
        print(f"❌ Error accessing Feature Store: {e}")
    
    # Check Model Registry
    print("\n" + "="*80)
    print("🤖 MODEL REGISTRY STATUS")
    print("="*80)
    
    try:
        mr = project.get_model_registry()
        
        # Check for models
        models_found = []
        for day in [1, 2, 3]:
            for model_type in ['catboost', 'ensemble', 'xgboost', 'lightgbm']:
                model_name = f"pearls_aqi_day{day}_{model_type}"
                try:
                    model = mr.get_model(model_name, version=1)
                    models_found.append({
                        'name': model_name,
                        'version': model.version,
                        'created': model.created
                    })
                except:
                    pass
        
        if models_found:
            print(f"\n✅ Found {len(models_found)} models in registry:\n")
            for m in models_found:
                print(f"   • {m['name']} (v{m['version']}) - Created: {m['created']}")
            print("\n✅ MODELS UPLOADED SUCCESSFULLY!")
        else:
            print("\n⚠️  No models found in registry")
            print("❌ MODELS NOT UPLOADED YET")
            
    except Exception as e:
        print(f"❌ Error accessing Model Registry: {e}")
    
    print("\n" + "="*80)
    print("🌐 Hopsworks Dashboard:")
    print(f"   https://c.app.hopsworks.ai/p/{project.id}")
    print("="*80)
    
except ImportError:
    print("\n❌ Hopsworks package not installed")
    print("   Install: pip install hopsworks")
except Exception as e:
    print(f"\n❌ Error connecting to Hopsworks: {e}")

print("\n" + "="*80)
print("VERIFICATION COMPLETE")
print("="*80)
