import fastf1
import pandas as pd
import shutil
import numpy as np
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import cross_val_score, LeaveOneOut
from sklearn.ensemble import VotingRegressor
import xgboost as xgb
import fastf1.req
from sklearn.preprocessing import StandardScaler


def run_predictions_for_grand_prix(grand_prix: str, current_year: int):
    # Cache location setup
    cache_location = "/content/f1_cache"
    if os.path.exists(cache_location):
        shutil.rmtree(cache_location)
    os.makedirs(cache_location, exist_ok=True)
    fastf1.Cache.enable_cache(cache_location)

    # Previous year data
    previous_year = current_year - 1

    # Load FastF1 sessions for the previous year
    session_2024 = fastf1.get_session(previous_year, grand_prix, "R")
    session_2024.load()
    
    # Load qualifying data for both years
    quali_2025 = fastf1.get_session(current_year, grand_prix, "Q")
    quali_2025.load()
    quali_2024 = fastf1.get_session(previous_year, grand_prix, "Q")
    quali_2024.load()

    # Prepare data for training
    quali24 = quali_2024.results[['DriverNumber', 'TeamName', 'Q1', 'Q2', 'Q3', 'Position', 'Abbreviation']].copy()
    session24 = session_2024.results[['DriverNumber', 'Time', 'Status']].copy()

    data = pd.merge(quali24, session24, on='DriverNumber', how='inner')

    max_laptime = session_2024.laps['LapTime'].max()
    data.loc[data['Status'] == 'Lapped', 'Time'] += max_laptime
    data = data.dropna(subset=['Time'])
    
    X_train = data[['TeamName', 'Q1', 'Q2', 'Q3', 'Position']].copy()
    y_train = data['Time'].dt.total_seconds()

    X_predict = quali_2025.results[['TeamName', 'Q1', 'Q2', 'Q3', 'Position']].copy()
    driver_abbr = X_predict['Abbreviation'].copy()
   
    
    # Preprocessing for training and prediction
    X_train = pd.get_dummies(X_train, columns=['TeamName'], prefix='Team')
    X_predict = pd.get_dummies(X_predict, columns=['TeamName'], prefix='Team')
    #X_predict.rename(columns={'Team_Racing Bulls': 'Team_RB'}, inplace=True)

    X_predict = X_predict.reindex(columns=X_train.columns, fill_value=0)

    X_train['Q2'] = X_train['Q2'].fillna(X_train['Q1'] + pd.Timedelta(seconds=10))
    X_train['Q3'] = X_train['Q3'].fillna(X_train['Q2'] + pd.Timedelta(seconds=10))
    X_train = X_train.drop(columns=['Abbreviation'])

    X_train.reset_index(drop=True, inplace=True)
    y_train.reset_index(drop=True, inplace=True)
    
    X_predict['Q2'] = X_predict['Q2'].fillna(X_predict['Q1'] + pd.Timedelta(seconds=10))
    X_predict['Q3'] = X_predict['Q3'].fillna(X_predict['Q2'] + pd.Timedelta(seconds=10))

    # Convert times to seconds for model training
    X_train['Q1'] = X_train['Q1'].dt.total_seconds()
    X_train['Q2'] = X_train['Q2'].dt.total_seconds()
    X_train['Q3'] = X_train['Q3'].dt.total_seconds()
    X_predict['Q1'] = X_predict['Q1'].dt.total_seconds()
    X_predict['Q2'] = X_predict['Q2'].dt.total_seconds()
    X_predict['Q3'] = X_predict['Q3'].dt.total_seconds()

    # Standardizing data
    scaler = StandardScaler()
    numerical_features = ['Q1', 'Q2', 'Q3', 'Position']
    X_train[numerical_features] = scaler.fit_transform(X_train[numerical_features])
    X_predict[numerical_features] = scaler.transform(X_predict[numerical_features])

    # Initialize models
    rf = RandomForestRegressor(n_estimators=100, random_state=10)
    xgb_model = xgb.XGBRegressor(n_estimators=100, random_state=10)
    gb = GradientBoostingRegressor(n_estimators=100, random_state=10)

    # Create ensemble
    ensemble = VotingRegressor([('rf', rf), ('gb', gb), ('xgb', xgb_model)])

    # Cross-validation using LeaveOneOut
    loo = LeaveOneOut()
    models = {'Random Forest': rf, 'Gradient Boosting': gb, 'XGBoost': xgb_model, 'Ensemble': ensemble}
    cv_results = {}

    for name, model in models.items():
        scores = cross_val_score(model, X_train, y_train['Time'], scoring='neg_mean_absolute_error', cv=loo)
        mae = -scores.mean()
        cv_results[name] = mae
        print(f'{name} CV MAE: {mae:.2f} seconds')

    # Feature importance analysis
    for name, model in models.items():
        if hasattr(model, 'feature_importances_'):
            print(f"\n{name} Feature Importances:")
            importances = pd.DataFrame({
                'feature': X_train.columns,
                'importance': model.feature_importances_
            })
            print(importances.sort_values('importance', ascending=False).head(10))

    # Train the models and make predictions
    predictions = {}
    for name, model in models.items():
        model.fit(X_train, y_train['Time'])
        pred_scaled = model.predict(X_predict)
        pred_adjusted = pred_scaled  # Adjust predictions if necessary
        
        # Save predictions
        predictions[name] = pred_adjusted
        
        results = pd.DataFrame({
            'Abbreviation': driver_abbr,
            'Predicted_Race_Time_sec': pred_adjusted,
            'Predicted_Race_Time': [seconds_to_time(s) for s in pred_adjusted]
        })
        
        results_sorted = results.sort_values('Predicted_Race_Time_sec')
        print(f"\n{name} Predictions:")
        print(results_sorted[['Abbreviation', 'Predicted_Race_Time']])

    # Determine winner
    winner_abb = results_sorted['Abbreviation'].iloc[0]
    winner_data = quali_2025.results[quali_2025.results['Abbreviation'] == winner_abb]
    winner_name = winner_data['FullName'].iloc[0]
    print(f"And the winner of the 2025 {grand_prix} Grand Prix is {winner_name}!")


def seconds_to_time(seconds):
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:06.3f}"


# Esegui per un gran premio specifico, esempio:
run_predictions_for_grand_prix(grand_prix='China', current_year=2025)
