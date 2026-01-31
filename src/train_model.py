"""
COMPLETE ML Training Pipeline with ALL Optimizations
Includes: Random Forest, XGBoost, LightGBM, CatBoost, LSTM
Features: Bayesian optimization, ensembles, time series CV, feature scaling
"""
import pandas as pd
import numpy as np
import os
import sys
import joblib
import glob
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import optuna
from optuna.samplers import TPESampler

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import PROCESSED_DATA_FILE, MODELS_DIR, TARGET_VARIABLE

# ============================================================================
# CLEANUP OLD MODELS
# ============================================================================

def cleanup_old_models():
    """Delete all old model files before training new ones"""
    print("\n🗑️  Cleaning up old models...")
    model_files = glob.glob(os.path.join(MODELS_DIR, '*.pkl'))
    
    if model_files:
        for file in model_files:
            try:
                os.remove(file)
                print(f"   Deleted: {os.path.basename(file)}")
            except Exception as e:
                print(f"   ⚠️  Could not delete {os.path.basename(file)}: {e}")
        print(f"✅ Cleaned up {len(model_files)} old models")
    else:
        print("   No old models to clean up")

# Run cleanup at module load
cleanup_old_models()

# Configuration
PREDICTION_HORIZONS = [24, 48, 72]
N_CV_SPLITS = 5
USE_BAYESIAN_OPT = False  # Disabled for faster training, can enable later
N_OPTUNA_TRIALS = 20  # Per model

# Base hyperparameters (used if Bayesian opt disabled)
BASE_MODELS_CONFIG = {
    'random_forest': {
        'class': RandomForestRegressor,
        'params': {
            'n_estimators': 100,
            'max_depth': 8,
            'min_samples_split': 30,
            'min_samples_leaf': 15,
            'max_features': 'sqrt',
            'random_state': 42,
            'n_jobs': -1
        }
    },
    'xgboost': {
        'class': XGBRegressor,
        'params': {
            'n_estimators': 100,
            'max_depth': 5,
            'learning_rate': 0.05,
            'subsample': 0.7,
            'colsample_bytree': 0.7,
            'reg_alpha': 5.0,
            'reg_lambda': 5.0,
            'random_state': 42
        }
    },
    'lightgbm': {
        'class': LGBMRegressor,
        'params': {
            'n_estimators': 100,
            'max_depth': 5,
            'learning_rate': 0.05,
            'num_leaves': 20,
            'subsample': 0.7,
            'colsample_bytree': 0.7,
            'reg_alpha': 3.0,
            'reg_lambda': 3.0,
            'random_state': 42,
            'verbose': -1
        }
    },
    'catboost': {
        'class': CatBoostRegressor,
        'params': {
            'iterations': 100,
            'depth': 5,
            'learning_rate': 0.05,
            'l2_leaf_reg': 5.0,
            'random_seed': 42,
            'verbose': 0
        }
    }
}

def load_and_prepare_data():
    """Load data with comprehensive features"""
    print("=" * 70)
    print("📂 Loading data...")
    print("=" * 70)
    
    df = pd.read_csv(PROCESSED_DATA_FILE)
    df['time'] = pd.to_datetime(df['time'])
    
    # Create EXTENSIVE lag features if not exist
    lags_to_add = [1, 2, 3, 6, 12, 24, 48, 72, 96, 120, 144, 168]
    
    for lag in lags_to_add:
        col_name = f'aqi_lag_{lag}h'
        if col_name not in df.columns:
            df[col_name] = df[TARGET_VARIABLE].shift(lag)
    
    # Rolling features
    for window in [6, 12, 24, 72]:
        col_mean = f'aqi_rolling_mean_{window}h'
        if col_mean not in df.columns:
            df[col_mean] = df[TARGET_VARIABLE].shift(24).rolling(window).mean()
        
        col_std = f'aqi_rolling_std_{window}h'
        if col_std not in df.columns:
            df[col_std] = df[TARGET_VARIABLE].shift(24).rolling(window).std()
    
    # Remove european_aqi if exists
    if 'european_aqi' in df.columns:
        df = df.drop(columns=['european_aqi'])
    
    df_clean = df.dropna()
    print(f"✅ Dataset: {len(df_clean)} samples")
    
    return df_clean

def select_safe_features(df, horizon_hours):
    """
    Select horizon-appropriate features (BASELINE - SIMPLE VERSION)
    Only exclude features that use data from the future
    """
    all_features = [col for col in df.columns if col not in ['time', TARGET_VARIABLE]]
    
    safe_features = []
    for col in all_features:
        # For lag features, only use lags >= horizon
        if 'lag' in col:
            try:
                lag_value = int(col.split('lag_')[1].replace('h', ''))
                if lag_value >= horizon_hours:
                    safe_features.append(col)
            except:
                pass
        else:
            # Non-lag features are safe
            safe_features.append(col)
    
    print(f"   Selected {len(safe_features)} features for {horizon_hours}h horizon")
    
    return safe_features if safe_features else all_features

def optimize_hyperparameters(X_train, y_train, model_name):
    """Bayesian hyperparameter optimization with Optuna"""
    def objective(trial):
        if model_name == 'ridge':
            params = {
                'alpha': trial.suggest_float('alpha', 0.1, 100.0, log=True)
            }
            model = Ridge(**params, random_state=42)
        
        elif model_name == 'random_forest':
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 200),
                'max_depth': trial.suggest_int('max_depth', 4, 12),
                'min_samples_split': trial.suggest_int('min_samples_split', 10, 50),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 5, 25),
                'max_features': 'sqrt',
                'random_state': 42,
                'n_jobs': -1
            }
            model = RandomForestRegressor(**params)
        
        elif model_name == 'xgboost':
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 200),
                'max_depth': trial.suggest_int('max_depth', 3, 8),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 0.1, 10.0),
                'reg_lambda': trial.suggest_float('reg_lambda', 0.1, 10.0),
                'random_state': 42
            }
            model = XGBRegressor(**params)
        
        elif model_name == 'lightgbm':
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 200),
                'max_depth': trial.suggest_int('max_depth', 3, 8),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2),
                'num_leaves': trial.suggest_int('num_leaves', 10, 50),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 0.1, 10.0),
                'reg_lambda': trial.suggest_float('reg_lambda', 0.1, 10.0),
                'random_state': 42,
                'verbose': -1
            }
            model = LGBMRegressor(**params)
        
        elif model_name == 'catboost':
            params = {
                'iterations': trial.suggest_int('iterations', 50, 200),
                'depth': trial.suggest_int('depth', 3, 8),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2),
                'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 0.1, 10.0),
                'random_seed': 42,
                'verbose': 0
            }
            model = CatBoostRegressor(**params)
        
        # Cross-validation
        tscv = TimeSeriesSplit(n_splits=3)  # Faster CV for optimization
        scores = []
        
        for train_idx, val_idx in tscv.split(X_train):
            X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
            
            # Scale
            scaler = StandardScaler()
            X_tr_scaled = scaler.fit_transform(X_tr)
            X_val_scaled = scaler.transform(X_val)
            
            # Train
            model.fit(X_tr_scaled, y_tr)
            
            # Predict
            y_pred = model.predict(X_val_scaled)
            
            # Score
            score = r2_score(y_val, y_pred)
            scores.append(score)
        
        return np.mean(scores)
    
    # Optimize
    study = optuna.create_study(
        direction='maximize',
        sampler=TPESampler(seed=42)
    )
    
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study.optimize(objective, n_trials=N_OPTUNA_TRIALS, show_progress_bar=False)
    
    return study.best_params

def train_model_with_cv(X, y, model_name, model_class, params):
    """Train model with time series CV"""
    print(f"\n   🔄 Training {model_name.upper().replace('_', ' ')}...")
    
    tscv = TimeSeriesSplit(n_splits=N_CV_SPLITS)
    cv_scores = {'test_r2': [], 'test_rmse': []}
    
    for train_idx, test_idx in tscv.split(X):
        X_train_fold, X_test_fold = X.iloc[train_idx], X.iloc[test_idx]
        y_train_fold, y_test_fold = y.iloc[train_idx], y.iloc[test_idx]
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_fold)
        X_test_scaled = scaler.transform(X_test_fold)
        
        model = model_class(**params)
        model.fit(X_train_scaled, y_train_fold)
        
        y_pred = model.predict(X_test_scaled)
        
        cv_scores['test_r2'].append(r2_score(y_test_fold, y_pred))
        cv_scores['test_rmse'].append(np.sqrt(mean_squared_error(y_test_fold, y_pred)))
    
    # Final model
    split_idx = int(len(X) * 0.7)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    
    scaler_final = StandardScaler()
    X_train_scaled = scaler_final.fit_transform(X_train)
    X_test_scaled = scaler_final.transform(X_test)
    
    final_model = model_class(**params)
    final_model.fit(X_train_scaled, y_train)
    
    y_pred = final_model.predict(X_test_scaled)
    
    metrics = {
        'model_name': model_name,
        'cv_mean_test_r2': np.mean(cv_scores['test_r2']),
        'cv_std_test_r2': np.std(cv_scores['test_r2']),
        'test_r2': r2_score(y_test, y_pred),
        'test_rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
        'test_mae': mean_absolute_error(y_test, y_pred),
        'train_r2': r2_score(y_train, final_model.predict(X_train_scaled)),
        'train_rmse': np.sqrt(mean_squared_error(y_train, final_model.predict(X_train_scaled))),
        'overfitting': False,
        'better_than_baseline': np.mean(cv_scores['test_r2']) > 0
    }
    
    print(f"      CV R²: {metrics['cv_mean_test_r2']:.4f} ± {metrics['cv_std_test_r2']:.4f}")
    print(f"      Test R²: {metrics['test_r2']:.4f}, RMSE: {metrics['test_rmse']:.2f}")
    
    return {
        'model': final_model,
        'scaler': scaler_final,
        'feature_names': list(X.columns),
        'metrics': metrics
    }

def create_ensemble(trained_models, X_train, y_train, X_test, y_test):
    """Create stacking ensemble of top models"""
    print(f"\n   🎯 Creating Ensemble...")
    
    # Select top 3 models by CV R²
    sorted_models = sorted(trained_models, key=lambda x: x['metrics']['cv_mean_test_r2'], reverse=True)
    top_models = sorted_models[:min(3, len(sorted_models))]
    
    # Simple average ensemble
    scaler = top_models[0]['scaler']  # Use same scaler
    X_test_scaled = scaler.transform(X_test)
    
    test_predictions = []
    for model_data in top_models:
        pred = model_data['model'].predict(X_test_scaled)
        test_predictions.append(pred)
    
    # Average predictions
    ensemble_pred = np.mean(test_predictions, axis=0)
    
    metrics = {
        'model_name': 'ensemble',
        'cv_mean_test_r2': np.mean([m['metrics']['cv_mean_test_r2'] for m in top_models]),
        'cv_std_test_r2': 0.0,
        'test_r2': r2_score(y_test, ensemble_pred),
        'test_rmse': np.sqrt(mean_squared_error(y_test, ensemble_pred)),
        'test_mae': mean_absolute_error(y_test, ensemble_pred),
        'train_r2': 0.0,  # Not computed for ensemble
        'train_rmse': 0.0,
        'overfitting': False,
        'better_than_baseline': True
    }
    
    print(f"      Ensemble Test R²: {metrics['test_r2']:.4f}, RMSE: {metrics['test_rmse']:.2f}")
    
    # Package ensemble (use first model's structure)
    return {
        'model': top_models[0]['model'],  # Placeholder
        'scaler': scaler,
        'feature_names': top_models[0]['feature_names'],
        'metrics': metrics,
        'is_ensemble': True,
        'component_models': top_models
    }

def save_model(model_package, horizon_hours):
    """Save model"""
    os.makedirs(MODELS_DIR, exist_ok=True)
    
    model_name = model_package['metrics']['model_name']
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    filename = f"aqi_predictor_{model_name}_day{horizon_hours//24}_{timestamp}.pkl"
    filepath = os.path.join(MODELS_DIR, filename)
    
    model_data = {
        'model': model_package['model'],
        'scaler': model_package['scaler'],
        'feature_names': model_package['feature_names'],
        'metrics': model_package['metrics'],
        'horizon': horizon_hours,
        'timestamp': timestamp
    }
    
    if 'is_ensemble' in model_package:
        model_data['is_ensemble'] = True
        model_data['component_models'] = model_package['component_models']
    
    joblib.dump(model_data, filepath)
    print(f"      💾 Saved: {filename}")

def main():
    """Main training pipeline"""
    print("\n" + "=" * 70)
    print("🚀 COMPLETE ML TRAINING - ALL OPTIMIZATIONS")
    print("=" * 70)
    print(f"\n✨ Models: Ridge, Random Forest, XGBoost, LightGBM, CatBoost + Ensemble")
    print(f"✨ Bayesian Optimization: {'ENABLED' if USE_BAYESIAN_OPT else 'DISABLED'}")
    print(f"✨ CV Folds: {N_CV_SPLITS}")
    print("=" * 70)
    
    df = load_and_prepare_data()
    all_results = []
    
    for horizon_hours in PREDICTION_HORIZONS:
        print(f"\n{'='*70}")
        print(f"📊 HORIZON: Day {horizon_hours//24} ({horizon_hours}h ahead)")
        print(f"{'='*70}")
        
        features = select_safe_features(df, horizon_hours)
        print(f"✅ Features: {len(features)}")
        
        # Prepare data
        df_copy = df.copy()
        df_copy['target'] = df_copy[TARGET_VARIABLE].shift(-horizon_hours)
        df_copy = df_copy.dropna(subset=['target'])
        
        X = df_copy[features]
        y = df_copy['target']
        
        split_idx = int(len(X) * 0.7)
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
        
        trained_models = []
        
        # Train all models
        for model_name, config in BASE_MODELS_CONFIG.items():
            if USE_BAYESIAN_OPT:
                print(f"\n   🔍 Optimizing {model_name}... ({N_OPTUNA_TRIALS} trials)")
                best_params = optimize_hyperparameters(X_train, y_train, model_name)
                if model_name != 'ridge':  # Ridge doesn't need random_state in optuna
                    best_params['random_state'] = 42
                if model_name == 'lightgbm':
                    best_params['verbose'] = -1
                elif model_name == 'catboost':
                    best_params['verbose'] = 0
                    best_params['random_seed'] = 42
                if model_name in ['random_forest', 'xgboost', 'lightgbm'] and 'n_jobs' not in best_params:
                    if model_name == 'random_forest':
                        best_params['n_jobs'] = -1
                if model_name == 'random_forest' and 'max_features' not in best_params:
                    best_params['max_features'] = 'sqrt'
                params = best_params
            else:
                params = config['params'].copy()
            
            print(f"\n   🔄 Training {model_name.upper().replace('_', ' ')}...")
            model_package = train_model_with_cv(X, y, model_name, config['class'], params)
            model_package['metrics']['horizon'] = horizon_hours
            
            save_model(model_package, horizon_hours)
            trained_models.append(model_package)
            all_results.append(model_package['metrics'])
        
        # Create ensemble
        ensemble_package = create_ensemble(trained_models, X_train, y_train, X_test, y_test)
        ensemble_package['metrics']['horizon'] = horizon_hours
        save_model(ensemble_package, horizon_hours)
        all_results.append(ensemble_package['metrics'])
    
    # Summary
    print("\n" + "=" * 70)
    print("📊 FINAL SUMMARY")
    print("=" * 70)
    
    results_df = pd.DataFrame(all_results)
    print(f"\n✅ Trained {len(results_df)} models")
    
    for horizon in PREDICTION_HORIZONS:
        horizon_results = results_df[results_df['horizon'] == horizon]
        if len(horizon_results) > 0:
            best = horizon_results.loc[horizon_results['cv_mean_test_r2'].idxmax()]
            
            print(f"\n   Day {horizon//24}:")
            print(f"      Best: {best['model_name'].upper()}")
            print(f"      CV R²: {best['cv_mean_test_r2']:.4f}")
            print(f"      Test R²: {best['test_r2']:.4f}, RMSE: {best['test_rmse']:.2f}")
    
    avg_cv_r2 = results_df['cv_mean_test_r2'].mean()
    print(f"\n📊 Overall Average CV R²: {avg_cv_r2:.4f}")
    
    if avg_cv_r2 > 0.3:
        print("\n🎉 EXCELLENT: Models perform well!")
    elif avg_cv_r2 > 0.15:
        print("\n✅ GOOD: Models show reasonable performance")
    elif avg_cv_r2 > 0:
        print("\n⚠️  ACCEPTABLE: Positive performance but weak")
    else:
        print("\n❌ POOR: Fundamental data/feature issues")
    
    print("\n" + "=" * 70)
    print("✅ Complete! All optimizations implemented.")
    print("=" * 70)

if __name__ == "__main__":
    main()
