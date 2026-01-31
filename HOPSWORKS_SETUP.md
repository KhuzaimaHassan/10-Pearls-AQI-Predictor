# Hopsworks Setup Guide

## ⚠️ Important: Configure Before Upload

To upload features and models to Hopsworks, follow these steps:

## Step 1: Create Hopsworks Account

1. Visit [https://app.hopsworks.ai](https://app.hopsworks.ai)
2. Click "Sign Up" (or "Login" if you have an account)
3. Complete registration with your email

## Step 2: Create a Project

1. After login, click "Create New Project"
2. Enter project name (e.g., `Pearls_AQI` or `aqi_predictor`)
3. Click "Create"
4. **IMPORTANT**: Note the exact project name (case-sensitive)

## Step 3: Generate API Key

1. Click on your profile icon (top right)
2. Go to "Account Settings"
3. Navigate to "API Keys" tab
4. Click "Generate New API Key"
5. Enter a name (e.g., `pearls-aqi-key`)
6. Click "Create"
7. **IMPORTANT**: Copy the API key immediately (you can't see it again!)

## Step 4: Configure Environment Variables

Edit the `.env` file in your project root:

```env
HOPSWORKS_API_KEY=your_copied_api_key_here
HOPSWORKS_PROJECT_NAME=Pearls_AQI
```

**Replace**:
- `your_copied_api_key_here` → Your actual API key from Step 3
- `Pearls_AQI` → Your actual project name from Step 2 (must match exactly!)

## Step 5: Upload to Hopsworks

Run the upload script:

```bash
# Activate virtual environment
.\.venv\Scripts\activate

# Upload features and models
python src\upload_to_hopsworks.py
```

## What Gets Uploaded?

### Features (Feature Store)
- **Feature Group Name**: `air_quality_karachi`
- **Records**: ~8,760 (1 year of hourly data)
- **Features**: 24 columns
  - Time series data
  - Lag features
  - Rolling statistics
  - Cyclical encodings
  - Weather data
  - Interaction features

### Models (Model Registry)
- **Total Models**: 9 (3 algorithms × 3 horizons)
- **Models**:
  - `aqi_predictor_linear_regression_day1`
  - `aqi_predictor_random_forest_day1`
  - `aqi_predictor_xgboost_day1`
  - `aqi_predictor_linear_regression_day2`
  - `aqi_predictor_random_forest_day2`
  - `aqi_predictor_xgboost_day2`
  - `aqi_predictor_linear_regression_day3`
  - `aqi_predictor_random_forest_day3`
  - `aqi_predictor_xgboost_day3`

Each model includes:
- Model file (.pkl)
- Performance metrics (RMSE, R²)
- Prediction horizon
- Model type

## Verify Upload

After uploading, verify in Hopsworks UI:

1. **Feature Store**:
   - Navigate to "Feature Store" in left menu
   - Look for `air_quality_karachi` feature group
   - Check the schema and preview data

2. **Model Registry**:
   - Navigate to "Model Registry" in left menu
   - Find your 9 models
   - Check metrics and metadata

## Troubleshooting

### Error: "API key not configured"
- **Solution**: Make sure `.env` file has correct API key

### Error: "Project not found"
- **Solution**: Verify project name matches exactly (case-sensitive)

### Error: "Connection failed"
- **Solution**: Check internet connection

### Error: "Feature group already exists"
- **Solution**: This is OK! The script will update existing data

### Error: "hopsworks package not installed"
- **Solution**: Run `pip install hopsworks`

## Benefits of Hopsworks

✅ **Feature Store**:
- Version control for features
- Time-travel queries
- Feature lineage tracking
- Shareable across teams

✅ **Model Registry**:
- Model versioning
- Experiment tracking
- A/B testing support
- Easy deployment

✅ **Automation**:
- GitHub Actions integration
- Scheduled feature updates
- Automated model retraining

## Next Steps After Upload

1. ✅ Verify data in Hopsworks UI
2. ✅ Set up GitHub repository
3. ✅ Configure GitHub Secrets for CI/CD
4. ✅ Enable GitHub Actions workflow
5. ✅ Deploy Streamlit app to cloud

---

**Note**: Hopsworks integration is **optional** but **recommended** for production deployments. The system works perfectly without it for local use!
