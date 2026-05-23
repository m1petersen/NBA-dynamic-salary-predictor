import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import VotingRegressor, RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.compose import TransformedTargetRegressor
from sklearn.metrics import r2_score, mean_absolute_error
import json

# 1. Load and Clean Data
df = pd.read_csv('NBA_Stats_and_Salaries_All_Seasons.csv')
df = df.dropna(subset=['Salary'])
df = df[df['GP'] >= 15]
df = df[df['Salary'] >= 500000]

# 2. Advanced Feature Engineering
df['MIN_SAFE'] = df['MIN'] + 1e-5
df['PTS_PER_MIN'] = df['PTS'] / df['MIN_SAFE']
df['FANTASY_PER_MIN'] = df['NBA_FANTASY_PTS'] / df['MIN_SAFE']
df['USAGE_PROXY'] = (df['FGA'] + 0.44 * df['FTA'] + df['TOV']) / df['MIN_SAFE']
df['TS_PROX'] = df['PTS'] / (2 * (df['FGA'] + 0.44 * df['FTA'] + 1e-5))

# Career Arc mapping
df['AGE_SQ'] = df['AGE'] ** 2
df['IS_ROOKIE_AGE'] = (df['AGE'] <= 24).astype(int)
df['VETERAN_PRIME'] = ((df['AGE'] >= 26) & (df['AGE'] <= 31)).astype(int)

features = [
    'AGE', 'AGE_SQ', 'IS_ROOKIE_AGE', 'VETERAN_PRIME', 'GP', 'W_PCT', 'MIN', 'PTS', 
    'REB', 'AST', 'STL', 'BLK', 'TOV', 'PLUS_MINUS', 'NBA_FANTASY_PTS', 
    'PTS_PER_MIN', 'FANTASY_PER_MIN', 'USAGE_PROXY', 'TS_PROX'
]
df = df.dropna(subset=features)

seasons_order = ['2022/2023', '2023/2024', '2024/2025']
results = {}

for i, season in enumerate(seasons_order):
    current_seasons = seasons_order[:i+1]
    df_sub = df[df['Season'].isin(current_seasons)]
    
    X = df_sub[features]
    y = df_sub['Salary']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)
    
    # 3. Log-Transformed Ensemble Architecture
    rf = RandomForestRegressor(n_estimators=400, max_depth=16, min_samples_leaf=2, random_state=42, n_jobs=-1)
    hgb = HistGradientBoostingRegressor(max_iter=400, max_depth=7, learning_rate=0.02, min_samples_leaf=5, l2_regularization=2.0, random_state=42)
    
    ensemble = VotingRegressor([('rf', rf), ('hgb', hgb)], weights=[1.0, 1.5])
    log_model = TransformedTargetRegressor(regressor=ensemble, func=np.log1p, inverse_func=np.expm1)
    
    log_model.fit(X_train, y_train)
    
    # Evaluate
    preds_test_raw = log_model.predict(X_test)
    r2 = r2_score(y_test, preds_test_raw)
    mae = mean_absolute_error(y_test, preds_test_raw)
    
    # Extract feature importance from RF component
    rf.fit(X_train, np.log1p(y_train))
    importances = rf.feature_importances_
    feat_imp = [{'feature': f, 'importance': float(imp)} for f, imp in zip(features, importances)]
    feat_imp = sorted(feat_imp, key=lambda x: x['importance'], reverse=True)[:6]
    
    # Predictions for UI
    df_sub_copy = df_sub.copy()
    df_sub_copy['Predicted_Salary'] = log_model.predict(X).clip(min=500000)
    df_sub_copy['Difference'] = df_sub_copy['Predicted_Salary'] - df_sub_copy['Salary']
    
    underpaid = df_sub_copy.sort_values(by='Difference', ascending=False).head(5)
    overpaid = df_sub_copy.sort_values(by='Difference', ascending=True).head(5)
    
    # CRITICAL: Added statistics directly into the scatter array format
    scatter_fields = [
        'PLAYER_NAME', 'Salary', 'Predicted_Salary', 'Season', 'Difference',
        'NBA_FANTASY_PTS', 'MIN', 'GP', 'AGE'
    ]
    
    results[f"step_{i+1}"] = {
        "seasons_label": " & ".join(current_seasons),
        "r2": round(float(r2), 3),
        "mae": round(float(mae), 2),
        "feature_importance": feat_imp,
        "underpaid": underpaid[['PLAYER_NAME', 'Salary', 'Predicted_Salary', 'Difference']].to_dict('records'),
        "overpaid": overpaid[['PLAYER_NAME', 'Salary', 'Predicted_Salary', 'Difference']].to_dict('records'),
        "scatter": df_sub_copy[scatter_fields].to_dict('records')
    }

with open('nba_seasons_data.json', 'w') as f:
    json.dump(results, f)

print("Dataset generated successfully with performance metrics attached!")