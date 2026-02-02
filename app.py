"""
Pearls AQI Predictor - Streamlit Dashboard
Beautiful, functional 3-day AQI forecast application
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import os
import joblib
from datetime import datetime, timedelta
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from src.config import (
    PROCESSED_DATA_FILE, 
    MODELS_DIR, 
    TARGET_VARIABLE,
    get_aqi_category,
    get_aqi_color
)

# Page configuration
st.set_page_config(
    page_title="Pearls AQI Predictor - Karachi Air Quality Forecast",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced custom CSS
st.markdown("""
<style>
    /* Main styling */
    .main {
        background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
        color: white;
    }
    
    /* Headers */
    .main-header {
        font-size: 3.5rem;
        font-weight: 800;
        text-align: center;
        background: linear-gradient(120deg, #00d4ff 0%, #0099ff 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
        text-shadow: 0 0 30px rgba(0, 212, 255, 0.3);
    }
    
    .sub-header {
        font-size: 1.3rem;
        text-align: center;
        color: #a0aec0;
        margin-bottom: 2rem;
    }
    
    /* Metric cards */
    .metric-card {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        border-radius: 15px;
        padding: 1.5rem;
        border: 1px solid rgba(255, 255, 255, 0.1);
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
        transition: all 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 40px rgba(0, 212, 255, 0.2);
        border-color: rgba(0, 212, 255, 0.3);
    }
    
    .metric-label {
        font-size: 0.9rem;
        color: #a0aec0;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 0.5rem;
    }
    
    .metric-value {
        font-size: 2.5rem;
        font-weight: 700;
        color: #00d4ff;
        margin: 0;
    }
    
    /* AQI badges */
    .aqi-good {
        background: linear-gradient(135deg, #00e676 0%, #00c853 100%);
        color: white;
        padding: 0.8rem 1.5rem;
        border-radius: 10px;
        font-weight: 600;
        font-size: 1.1rem;
        box-shadow: 0 4px 15px rgba(0, 230, 118, 0.4);
    }
    
    .aqi-moderate {
        background: linear-gradient(135deg, #ffd600 0%, #ffab00 100%);
        color: #1a1a1a;
        padding: 0.8rem 1.5rem;
        border-radius: 10px;
        font-weight: 600;
        font-size: 1.1rem;
        box-shadow: 0 4px 15px rgba(255, 214, 0, 0.4);
    }
    
    .aqi-unhealthy-for-sensitive {
        background: linear-gradient(135deg, #ff9100 0%, #ff6d00 100%);
        color: white;
        padding: 0.8rem 1.5rem;
        border-radius: 10px;
        font-weight: 600;
        font-size: 1.1rem;
        box-shadow: 0 4px 15px rgba(255, 145, 0, 0.4);
    }
    
    .aqi-unhealthy {
        background: linear-gradient(135deg, #f44336 0%, #d32f2f 100%);
        color: white;
        padding: 0.8rem 1.5rem;
        border-radius: 10px;
        font-weight: 600;
        font-size: 1.1rem;
        box-shadow: 0 4px 15px rgba(244, 67, 54, 0.4);
    }
    
    .aqi-very-unhealthy {
        background: linear-gradient(135deg, #9c27b0 0%, #7b1fa2 100%);
        color: white;
        padding: 0.8rem 1.5rem;
        border-radius: 10px;
        font-weight: 600;
        font-size: 1.1rem;
        box-shadow: 0 4px 15px rgba(156, 39, 176, 0.4);
    }
    
    .aqi-hazardous {
        background: linear-gradient(135deg, #b71c1c 0%, #880e4f 100%);
        color: white;
        padding: 0.8rem 1.5rem;
        border-radius: 10px;
        font-weight: 600;
        font-size: 1.1rem;
        box-shadow: 0 4px 15px rgba(183, 28, 28, 0.4);
    }
    
    /* Success message */
    .stSuccess {
        background: rgba(0, 230, 118, 0.1);
        border-left: 4px solid #00e676;
    }
    
    /* Sidebar styling */
    .css-1d391kg, [data-testid="stSidebar"] {
        background: rgba(15, 12, 41, 0.95);
        backdrop-filter: blur(10px);
    }
    
    /* Buttons */
    .stButton>button {
        background: linear-gradient(135deg, #00d4ff 0%, #0099ff 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.6rem 1.5rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        transform: scale(1.05);
        box-shadow: 0 6px 20px rgba(0, 212, 255, 0.4);
    }
    
    /* Info boxes */
    .info-box {
        background: rgba(0, 212, 255, 0.1);
        border-left: 4px solid #00d4ff;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data(ttl=300)  # Cache for 5 minutes (reduced from 1 hour)
def load_data():
    """
    Load historical data from Hopsworks Feature Store
    Falls back to local files if Hopsworks not available
    """
    try:
        # Try to get credentials from environment variables (HF Spaces) or st.secrets (Streamlit Cloud) or .env (local)
        api_key = None
        project_name = None
        
        # Priority 1: Environment variables (Hugging Face Spaces)
        if os.getenv('HOPSWORKS_API_KEY'):
            api_key = os.getenv('HOPSWORKS_API_KEY')
            project_name = os.getenv('HOPSWORKS_PROJECT_NAME', 'three_days_AQI')
            st.info("🔑 Using Hopsworks credentials from environment variables")
        # Priority 2: Streamlit secrets (Streamlit Cloud)
        elif hasattr(st, 'secrets') and 'HOPSWORKS_API_KEY' in st.secrets:
            api_key = st.secrets['HOPSWORKS_API_KEY']
            project_name = st.secrets.get('HOPSWORKS_PROJECT_NAME', 'three_days_AQI')
            st.info("🔑 Using Hopsworks credentials from Streamlit secrets")
        # Priority 3: .env file (local development)
        elif os.path.exists('.env'):
            from dotenv import load_dotenv
            load_dotenv()
            api_key = os.getenv('HOPSWORKS_API_KEY')
            project_name = os.getenv('HOPSWORKS_PROJECT_NAME', 'three_days_AQI')
            st.info("🔑 Using Hopsworks credentials from .env file")
            
        if api_key:
            import hopsworks
            project = hopsworks.login(api_key_value=api_key, project=project_name)
            fs = project.get_feature_store()
            
            # Get feature group and read from offline storage (bypass query service)
            fg = fs.get_feature_group(name="aqi_features", version=1)
            
            # Use offline storage with explicit read options to avoid query service
            df = fg.read(online=False, read_options={"use_hive": True})
            
            
            if df is not None and not df.empty:
                df['time'] = pd.to_datetime(df['time'])
                df = df.sort_values('time')
                # Data loaded successfully (silent)
                return df
            else:
                st.warning("Feature group exists but is empty")
                
    except Exception as e:
        st.warning(f"Could not load from Hopsworks: {str(e)}")
    
    # Fallback to local files
    if not os.path.exists(PROCESSED_DATA_FILE):
        st.error("❌ No data available. Please configure Hopsworks secrets.")
        return None
        
    df = pd.read_csv(PROCESSED_DATA_FILE)
    df['time'] = pd.to_datetime(df['time'])
    st.info(f"📁 Loaded {len(df):,} records from local files")
    return df

@st.cache_resource
def load_models():
    """
    Load models from Hopsworks Model Registry
    Falls back to local files if Hopsworks not available
    """
    try:
        # Get credentials (same priority as load_data)
        api_key = None
        project_name = None
        
        # Priority 1: Environment variables (Hugging Face Spaces)
        if os.getenv('HOPSWORKS_API_KEY'):
            api_key = os.getenv('HOPSWORKS_API_KEY')
            project_name = os.getenv('HOPSWORKS_PROJECT_NAME', 'three_days_AQI')
        # Priority 2: Streamlit secrets (Streamlit Cloud)
        elif hasattr(st, 'secrets') and 'HOPSWORKS_API_KEY' in st.secrets:
            api_key = st.secrets['HOPSWORKS_API_KEY']
            project_name = st.secrets.get('HOPSWORKS_PROJECT_NAME', 'three_days_AQI')
        # Priority 3: .env file (local development)
        elif os.path.exists('.env'):
            from dotenv import load_dotenv
            load_dotenv()
            api_key = os.getenv('HOPSWORKS_API_KEY')
            project_name = os.getenv('HOPSWORKS_PROJECT_NAME', 'three_days_AQI')
            
        if api_key:
            import hopsworks
            import glob as glob_module
            project = hopsworks.login(api_key_value=api_key, project=project_name)
            mr = project.get_model_registry()
            
            models = {}
            model_types = ['catboost', 'xgboost', 'lightgbm', 'random_forest', 'ensemble']
            
            for day in [1, 2, 3]:
                for model_type in model_types:
                    try:
                        model_name = f"pearls_aqi_day{day}_{model_type}"
                        model = mr.get_model(model_name, version=None)  # Get latest version
                        model_dir = model.download()
                        model_files = glob_module.glob(f"{model_dir}/*.pkl")
                        
                        if model_files:
                            model_data = joblib.load(model_files[0])
                            models[f'{model_type}_day{day}'] = {
                                'model_data': model_data,
                                'model_type': model_type,
                                'day': day,
                                'name': model_name
                            }
                            # Model loaded successfully (silent)
                            pass
                    except Exception as e:
                        # Failed to load model (silent - just skip)
                        continue
            return models
    except Exception as e:
        st.warning(f"Could not load from Hopsworks: {e}. Trying local files...")
    
    # Fallback to local files
    if not os.path.exists(MODELS_DIR):
        return {}
    
    models = {}
    for filename in os.listdir(MODELS_DIR):
        if filename.endswith('.pkl'):
            filepath = os.path.join(MODELS_DIR, filename)
            try:
                model_data = joblib.load(filepath)
                parts = filename.replace('.pkl', '').split('_')
                
                if 'ensemble' in filename:
                    model_type = 'ensemble'
                    day_part = [p for p in parts if 'day' in p][0]
                elif 'random' in filename and 'forest' in filename:
                    model_type = 'random_forest'
                    day_part = [p for p in parts if 'day' in p][0]
                elif 'xgboost' in filename:
                    model_type = 'xgboost'
                    day_part = [p for p in parts if 'day' in p][0]
                elif 'lightgbm' in filename:
                    model_type = 'lightgbm'
                    day_part = [p for p in parts if 'day' in p][0]
                elif 'catboost' in filename:
                    model_type = 'catboost'
                    day_part = [p for p in parts if 'day' in p][0]
                else:
                    continue
                
                day = int(day_part.replace('day', ''))
                key = f"{model_type}_day{day}"
                models[key] = {
                    'model_data': model_data,
                    'model_type': model_type,
                    'day': day,
                    'metrics': model_data.get('metrics', {})
                }
            except Exception as e:
                continue
    
    return models

def get_best_model_for_day(models, day):
    """Get best performing model for a specific day"""
    candidate_models = {k: v for k, v in models.items() if v['day'] == day}
    
    if not candidate_models:
        return None
    
    # Try to get model with best metrics, otherwise just use CatBoost or first available
    best_key = None
    best_score = -999
    
    for key, model_dict in candidate_models.items():
        # Check if metrics exist in model_data
        model_data = model_dict.get('model_data', {})
        metrics = model_data.get('metrics', {})
        score = metrics.get('cv_mean_test_r2', -999)
        
        if score > best_score:
            best_score = score
            best_key = key
    
    # If no metrics found, prefer catboost, then first available
    if best_key is None or best_score == -999:
        if any('catboost' in k for k in candidate_models.keys()):
            best_key = [k for k in candidate_models.keys() if 'catboost' in k][0]
        else:
            best_key = list(candidate_models.keys())[0]
    
    return candidate_models[best_key]

def prepare_latest_features(df):
    """Prepare features from latest data point as a dictionary"""
    latest_row = df.iloc[-1]
    
    # Create feature dictionary
    features_dict = {}
    for col in df.columns:
        if col not in ['time', TARGET_VARIABLE]:
            features_dict[col] = latest_row[col]
    
    return features_dict

def make_prediction(model_data_dict, features_dict):
    """Make prediction using model with proper feature handling"""
    model_package = model_data_dict['model_data']
    model = model_package['model']
    
    # Check if model has scaler and feature names (new format)
    if 'scaler' in model_package and 'feature_names' in model_package:
        scaler = model_package['scaler']
        feature_names = model_package['feature_names']
        
        # Extract only the features the model was trained on, in correct order
        features_list = [features_dict.get(fname, 0) for fname in feature_names]
        
        # Scale features
        features_scaled = scaler.transform([features_list])
        
        # Predict
        prediction = model.predict(features_scaled)[0]
    else:
        # Old format - assume features_dict values match model expectations
        features_list = list(features_dict.values())
        prediction = model.predict([features_list])[0]
    
    return max(0, prediction)  # Ensure non-negative

def create_forecast_chart(predictions, day_labels=None):
    """Create beautiful 3-day forecast chart"""
    if day_labels is None:
        day_labels = ['Day 1 (24h)', 'Day 2 (48h)', 'Day 3 (72h)']
    
    colors = [get_aqi_color(pred) for pred in predictions]
    
    fig = go.Figure()
    
    # Add bars with gradient effect
    fig.add_trace(go.Bar(
        x=day_labels,
        y=predictions,
        marker=dict(
            color=colors,
            line=dict(color='rgba(255, 255, 255, 0.3)', width=2)
        ),
        text=[f"<b>{int(p)}</b>" for p in predictions],
        textposition='outside',
        textfont=dict(size=16, color='white'),
        hovertemplate='<b>%{x}</b><br>AQI: %{y:.0f}<extra></extra>'
    ))
    
    fig.update_layout(
        title=dict(
            text="<b>3-Day AQI Forecast</b>",
            font=dict(size=24, color='white'),
            x=0.5,
            xanchor='center'
        ),
        xaxis=dict(
            title=dict(text="<b>Forecast Period</b>", font=dict(size=14, color='#a0aec0')),
            tickfont=dict(size=12, color='white'),
            showgrid=False
        ),
        yaxis=dict(
            title=dict(text="<b>US AQI</b>", font=dict(size=14, color='#a0aec0')),
            tickfont=dict(size=12, color='white'),
            gridcolor='rgba(255, 255, 255, 0.1)'
        ),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        height=450,
        showlegend=False,
        hovermode='x unified',
        margin=dict(t=80, b=60, l=60, r=40)
    )
    
    return fig

def create_historical_chart(df, days=30):
    """Create historical AQI trend chart"""
    recent_df = df.tail(days * 24)
    
    fig = go.Figure()
    
    # Add AQI line
    fig.add_trace(go.Scatter(
        x=recent_df['time'],
        y=recent_df[TARGET_VARIABLE],
        mode='lines',
        name='US AQI',
        line=dict(color='#00d4ff', width=3),
        fill='tozeroy',
        fillcolor='rgba(0, 212, 255, 0.1)',
        hovertemplate='<b>%{x}</b><br>AQI: %{y:.0f}<extra></extra>'
    ))
    
    # Add reference lines
    fig.add_hline(y=50, line_dash="dash", line_color="yellow", 
                   annotation_text="Moderate", annotation_position="right")
    fig.add_hline(y=100, line_dash="dash", line_color="orange",
                   annotation_text="Unhealthy for Sensitive", annotation_position="right")
    fig.add_hline(y=150, line_dash="dash", line_color="red",
                   annotation_text="Unhealthy", annotation_position="right")
    
    fig.update_layout(
        title=dict(
            text=f"<b>AQI Trend - Last {days} Days</b>",
            font=dict(size=24, color='white'),
            x=0.5,
            xanchor='center'
        ),
        xaxis=dict(
            title=dict(text="<b>Date</b>", font=dict(size=14, color='#a0aec0')),
            tickfont=dict(size=11, color='white'),
            showgrid=False
        ),
        yaxis=dict(
            title=dict(text="<b>US AQI</b>", font=dict(size=14, color='#a0aec0')),
            tickfont=dict(size=12, color='white'),
            gridcolor='rgba(255, 255, 255, 0.1)'
        ),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        height=450,
        hovermode='x unified',
        margin=dict(t=80, b=60, l=60, r=40)
    )
    
    return fig

def display_aqi_badge(aqi_value):
    """Display AQI category badge"""
    category_name, color = get_aqi_category(aqi_value)
    color_class = category_name.lower().replace(' ', '-')
    
    health_messages = {
        'Good': 'Air quality is satisfactory, and air pollution poses little or no risk',
        'Moderate': 'Air quality is acceptable for most. Sensitive individuals should limit prolonged outdoor exertion',
        'Unhealthy For Sensitive Groups': 'Members of sensitive groups may experience health effects',
        'Unhealthy': 'Everyone may begin to experience health effects',
        'Very Unhealthy': 'Health alert: everyone may experience more serious health effects',
        'Hazardous': 'Health warnings of emergency conditions. Everyone is likely to be affected'
    }
    
    health_message = health_messages.get(category_name, 'Air quality status')
    
    st.markdown(f"""
    <div class="aqi-{color_class}">
        <div style="font-size: 1.2rem; font-weight: 700; margin-bottom: 0.3rem;">{category_name}</div>
        <div style="font-size: 0.9rem; opacity: 0.9;">{health_message}</div>
    </div>
    """, unsafe_allow_html=True)

def main():
    """Main dashboard application"""
    
    # Header
    st.markdown('<div class="main-header">🌍 Pearls AQI Predictor</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">3-Day Air Quality Forecast for Karachi, Pakistan</div>', unsafe_allow_html=True)
    
    # Load data
    with st.spinner('🔄 Loading data and models...'):
        df = load_data()
        models = load_models()
    
    if df is None:
        st.error("❌ No historical data found. Please run `python src/fetch_data.py` and `python src/feature_pipeline.py` first.")
        return
    
    if not models:
        st.error("❌ No trained models found. Please run `python src/train_model.py` first.")
        return
    
    st.success(f"✅ Loaded {len(df):,} historical records and {len(models)} trained models")
    
    # Sidebar
    st.sidebar.title("📊 Dashboard Controls")
    st.sidebar.markdown("---")
    
    # Model selection mode
    model_mode = st.sidebar.radio(
        "**Model Selection Mode**",
        ["Auto (Best Performance)", "Manual Selection"],
        help="Auto mode selects the best CV R² model for each day"
    )
    
    selected_models = {}
    
    if model_mode == "Manual Selection":
        st.sidebar.markdown("### Select Models")
        
        # Get unique model types
        model_types = sorted(list(set(m['model_type'] for m in models.values())))
        
        # Friendly names
        model_display_names = {
            'ridge': 'Ridge Regression',
            'random_forest': 'Random Forest',
            'xgboost': 'XGBoost',
            'lightgbm': 'LightGBM',
            'catboost': 'CatBoost',
            'ensemble': 'Ensemble (Top 3 Average)'
        }
        
        for day in [1, 2, 3]:
            available_models = [m['model_type'] for m in models.values() if m['day'] == day]
            display_options = [model_display_names.get(m, m.title()) for m in available_models]
            
            if available_models:
                selected_display = st.sidebar.selectbox(
                    f"Day {day}",
                    display_options,
                    key=f"model_day{day}"
                )
                
                # Map back to model type
                selected_type = available_models[display_options.index(selected_display)]
                key = f"{selected_type}_day{day}"
                selected_models[day] = models[key]
    else:
        # Auto mode - select best model for each day
        for day in [1, 2, 3]:
            best_model = get_best_model_for_day(models, day)
            if best_model:
                selected_models[day] = best_model
    
    # Sidebar - Model Status
    st.sidebar.markdown("---")
    st.sidebar.markdown("### ✅ System Status")
    
    # Check which model types are fully loaded (all 3 days)
    model_types_status = {}
    for model_type in ['catboost', 'xgboost', 'lightgbm', 'random_forest', 'ensemble']:
        days_loaded = [day for day in [1, 2, 3] if f'{model_type}_day{day}' in models]
        model_types_status[model_type] = len(days_loaded) == 3
    
    # Display model status
    model_display = {
        'catboost': 'CatBoost',
        'xgboost': 'XGBoost', 
        'lightgbm': 'LightGBM',
        'random_forest': 'Random Forest',
        'ensemble': 'Ensemble'
    }
    
    for model_type, is_loaded in model_types_status.items():
        icon = "✅" if is_loaded else "❌"
        st.sidebar.markdown(f"{icon} **{model_display[model_type]}**")
    
    st.sidebar.markdown(f"\n📊 **{len(df):,}** records loaded")
    
    # Add data freshness info
    if df is not None and not df.empty:
        latest_time = pd.to_datetime(df['time'].max())
        # Remove timezone info to avoid tz-naive/tz-aware comparison errors
        if latest_time.tz is not None:
            latest_time = latest_time.tz_localize(None)
        
        current_time = pd.Timestamp.now().tz_localize(None) if pd.Timestamp.now().tz else pd.Timestamp.now()
        hours_old = (current_time - latest_time).total_seconds() / 3600
        
        st.sidebar.markdown(f"**🕑 Data as of:** {latest_time.strftime('%Y-%m-%d %H:%M')}")
        
        if hours_old > 2:
            st.sidebar.warning(f"⚠️ Data is {hours_old:.1f} hours old")
        else:
            st.sidebar.success(f"✅ Fresh data ({hours_old:.1f}h old)")
    
    # Add manual refresh button
    st.sidebar.markdown("---")
    if st.sidebar.button("🔄 Refresh Data", help="Clear cache and reload from Hopsworks"):
        st.cache_data.clear()
        st.rerun()
    
    
    # Main tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📈 Forecast", "📊 Historical Trends", "🤖 Model Performance", "ℹ️ About"])
    
    # TAB 1: FORECAST
    with tab1:
        st.markdown("## 3-Day AQI Forecast")
        
        try:
            # Prepare features
            features_dict = prepare_latest_features(df)
            
            # Make predictions for 3 days
            predictions = []
            model_names = []
            
            for day in [1, 2, 3]:
                if day in selected_models:
                    pred = make_prediction(selected_models[day], features_dict)
                    predictions.append(pred)
                    model_names.append(selected_models[day]['model_type'].replace('_', ' ').title())
                else:
                    predictions.append(None)
                    model_names.append("N/A")
            
            # Forecast chart
            if all(p is not None for p in predictions):
                st.plotly_chart(create_forecast_chart(predictions), use_container_width=True)
                
                # Display predictions in cards
                st.markdown("### Detailed Predictions")
                cols = st.columns(3)
                
                for idx, (day, pred, model_name) in enumerate(zip([1, 2, 3], predictions, model_names)):
                    with cols[idx]:
                        st.markdown(f"""
                        <div class="metric-card">
                            <div class="metric-label">Day {day} ({24*day}h ahead)</div>
                            <div class="metric-value">{int(pred)}</div>
                            <div style="color: #a0aec0; font-size: 0.85rem; margin-top: 0.5rem;">
                                Model: {model_name}
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        st.markdown("<br>", unsafe_allow_html=True)
                        display_aqi_badge(pred)
            else:
                st.error("❌ Could not generate predictions for all 3 days. Check model availability.")
            
            # Current conditions
            st.markdown("---")
            st.markdown("## Current Conditions")
            
            latest = df.iloc[-1]
            # Get the actual latest timestamp from all data (not just last row)
            latest_timestamp = pd.to_datetime(df['time'].max())
            # Remove timezone for consistent display
            if latest_timestamp.tz is not None:
                latest_timestamp = latest_timestamp.tz_localize(None)
            current_time = latest_timestamp.strftime("%Y-%m-%d %H:%M")

            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Current AQI</div>
                    <div class="metric-value">{int(latest[TARGET_VARIABLE])}</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                pm25 = latest.get('pm2_5', 0)
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">PM2.5</div>
                    <div class="metric-value" style="font-size: 2rem;">{pm25:.1f}</div>
                    <div style="color: #a0aec0; font-size: 0.8rem;">μg/m³</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                pm10 = latest.get('pm10', 0)
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">PM10</div>
                    <div class="metric-value" style="font-size: 2rem;">{pm10:.1f}</div>
                    <div style="color: #a0aec0; font-size: 0.8rem;">μg/m³</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col4:
                temp = latest.get('temperature_2m', 0)
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Temperature</div>
                    <div class="metric-value" style="font-size: 2rem;">{temp:.1f}°C</div>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown(f'<p style="text-align: center; color: #a0aec0; margin-top: 1rem;">Last updated: {current_time}</p>', unsafe_allow_html=True)
            
        except Exception as e:
            st.error(f"❌ Error generating forecast: {str(e)}")
            st.exception(e)
    
    # TAB 2: HISTORICAL TRENDS
    with tab2:
        st.markdown("## Historical AQI Trends")
        
        days_to_show = st.slider("Select time range (days)", 7, 90, 30)
        
        st.plotly_chart(create_historical_chart(df, days_to_show), use_container_width=True)
        
        # Statistics
        recent = df.tail(days_to_show * 24)
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            avg_aqi = recent[TARGET_VARIABLE].mean()
            st.metric("Average AQI", f"{avg_aqi:.1f}")
        
        with col2:
            max_aqi = recent[TARGET_VARIABLE].max()
            st.metric("Maximum AQI", f"{max_aqi:.0f}")
        
        with col3:
            min_aqi = recent[TARGET_VARIABLE].min()
            st.metric("Minimum AQI", f"{min_aqi:.0f}")
        
        with col4:
            std_aqi = recent[TARGET_VARIABLE].std()
            st.metric("Std Deviation", f"{std_aqi:.1f}")
    
    # TAB 3: MODEL PERFORMANCE
    with tab3:
        st.markdown("## Model Performance Comparison")
        
        # Create comparison dataframe
        comparison_data = []
        for key, model_dict in models.items():
            # Safely get metrics from model_data
            model_data = model_dict.get('model_data', {})
            metrics = model_data.get('metrics', {})
            
            comparison_data.append({
                'Model': model_dict['model_type'].replace('_', ' ').title(),
                'Day': model_dict['day'],
                'CV R²': metrics.get('cv_mean_test_r2', 0),
                'Test R²': metrics.get('test_r2', 0),
                'RMSE': metrics.get('test_rmse', 0),
                'MAE': metrics.get('test_mae', 0)
            })
        
        comp_df = pd.DataFrame(comparison_data).sort_values(['Day', 'CV R²'], ascending=[True, False])
        
        # Display table
        st.dataframe(
            comp_df.style.format({
                'CV R²': '{:.4f}',
                'Test R²': '{:.4f}',
                'RMSE': '{:.2f}',
                'MAE': '{:.2f}'
            }).background_gradient(subset=['CV R²'], cmap='RdYlGn'),
            use_container_width=True,
            height=400
        )
        
        # Best models
        st.markdown("### 🏆 Best Models by Horizon")
        
        cols = st.columns(3)
        for idx, day in enumerate([1, 2, 3]):
            day_models = comp_df[comp_df['Day'] == day]
            if len(day_models) > 0:
                best = day_models.iloc[0]
                
                with cols[idx]:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-label">Day {day} Champion</div>
                        <div style="font-size: 1.5rem; font-weight: 700; color: #00d4ff; margin: 0.5rem 0;">
                            {best['Model']}
                        </div>
                        <div style="color: #a0aec0; font-size: 0.9rem;">
                            CV R²: {best['CV R²']:.4f}<br>
                            RMSE: {best['RMSE']:.2f}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
    
    # TAB 4: ABOUT
    with tab4:
        st.markdown("## About This Application")
        st.markdown("""
        This application provides a 3-day Air Quality Index (AQI) forecast for Karachi, Pakistan,
        along with historical trend analysis and model performance comparison.
        
        ### Data Source
        The historical and current weather data, including PM2.5, PM10, and temperature,
        is sourced from [Open-Meteo](https://open-meteo.com/).
        
        ### Forecasting Models
        The application utilizes an ensemble of machine learning models, including:
        - **CatBoost**
        - **XGBoost**
        - **LightGBM**
        - **Random Forest**
        
        These models are trained daily on the latest available data to provide accurate predictions.
        The 'Ensemble' model combines predictions from the individual models.
        
        ### AQI Calculation
        The AQI is calculated based on the concentration of various pollutants, primarily PM2.5 and PM10,
        following standard environmental agency guidelines.
        
        ### Development
        This application is built using Python and the Streamlit framework,
        making it easy to deploy interactive data science applications.
        
        ### Contact
        For any inquiries or feedback, please contact [your_email@example.com].
        """)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #a0aec0; padding: 1rem;">
        <p>Made with ❤️ using Streamlit | Data from Open-Meteo | Models trained daily</p>
        <p style="font-size: 0.85rem;">Location: Karachi, Pakistan (24.8607°N, 67.0011°E)</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
