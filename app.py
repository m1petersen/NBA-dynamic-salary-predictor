from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import VotingRegressor, RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.compose import TransformedTargetRegressor
import json

app = Flask(__name__)

# --- SYSTEM INIT: TRAIN ENSEMBLE PIPELINE ---
print("Initializing Analytics Engine & Training Predictive Models...")

df = pd.read_csv('NBA_Stats_and_Salaries_All_Seasons.csv').dropna(subset=['Salary'])
df = df[df['GP'] >= 15]
df = df[df['Salary'] >= 500000]

# Feature Engineering Formulas
df['MIN_SAFE'] = df['MIN'] + 1e-5
df['PTS_PER_MIN'] = df['PTS'] / df['MIN_SAFE']
df['FANTASY_PER_MIN'] = df['NBA_FANTASY_PTS'] / df['MIN_SAFE']
df['USAGE_PROXY'] = (df['FGA'] + 0.44 * df['FTA'] + df['TOV']) / df['MIN_SAFE']
df['TS_PROX'] = df['PTS'] / (2 * (df['FGA'] + 0.44 * df['FTA'] + 1e-5))
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
ui_static_data = {}
live_ensemble_model = None

# Iterative timeline assembly loop 
for i, season in enumerate(seasons_order):
    current_seasons = seasons_order[:i+1]
    df_sub = df[df['Season'].isin(current_seasons)]
    
    X = df_sub[features]
    y = df_sub['Salary']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)
    
    rf = RandomForestRegressor(n_estimators=400, max_depth=16, min_samples_leaf=2, random_state=42, n_jobs=-1)
    hgb = HistGradientBoostingRegressor(max_iter=400, max_depth=7, learning_rate=0.02, min_samples_leaf=5, l2_regularization=2.0, random_state=42)
    
    ensemble = VotingRegressor([('rf', rf), ('hgb', hgb)], weights=[1.0, 1.5])
    log_model = TransformedTargetRegressor(regressor=ensemble, func=np.log1p, inverse_func=np.expm1)
    log_model.fit(X_train, y_train)
    
    # Store complete 3-season trained state to power live user sandbox requests
    if i == 2:
        live_ensemble_model = log_model
        
    from sklearn.metrics import r2_score, mean_absolute_error
    preds_test = log_model.predict(X_test)
    r2 = r2_score(y_test, preds_test)
    mae = mean_absolute_error(y_test, preds_test)
    
    rf.fit(X_train, np.log1p(y_train))
    importances = rf.feature_importances_
    feat_imp = [{'feature': f, 'importance': float(imp)} for f, imp in zip(features, importances)]
    feat_imp = sorted(feat_imp, key=lambda x: x['importance'], reverse=True)[:6]
    
    df_sub_copy = df_sub.copy()
    df_sub_copy['Predicted_Salary'] = log_model.predict(X).clip(min=500000)
    df_sub_copy['Difference'] = df_sub_copy['Predicted_Salary'] - df_sub_copy['Salary']
    
    underpaid = df_sub_copy.sort_values(by='Difference', ascending=False).head(5)
    overpaid = df_sub_copy.sort_values(by='Difference', ascending=True).head(5)
    
    ui_static_data[f"step_{i+1}"] = {
        "seasons_label": " & ".join(current_seasons),
        "r2": round(float(r2), 3),
        "mae": round(float(mae), 2),
        "feature_importance": feat_imp,
        "underpaid": underpaid[['PLAYER_NAME', 'Salary', 'Predicted_Salary', 'Difference']].to_dict('records'),
        "overpaid": overpaid[['PLAYER_NAME', 'Salary', 'Predicted_Salary', 'Difference']].to_dict('records'),
        "scatter": df_sub_copy[['PLAYER_NAME', 'Salary', 'Predicted_Salary', 'Season', 'Difference']].to_dict('records')
    }

print("System Model Training Complete. Ready for UI requests.")

# --- ROUTING FRAMEWORK ---
@app.route('/')
def home():
    with open('index.html', 'r') as f:
        return f.read()

@app.route('/api/data')
def get_dashboard_data():
    return jsonify(ui_static_data)

@app.route('/api/predict', methods=['POST'])
def predict_custom_player():
    stats = request.json
    try:
        # Pull raw interactive telemetry parameters safely
        age = float(stats.get('AGE', 26))
        gp = float(stats.get('GP', 65))
        w_pct = float(stats.get('W_PCT', 50)) / 100.0  # Transform percentage to match fraction scale
        min_val = float(stats.get('MIN', 30))
        pts = float(stats.get('PTS', 15))
        reb = float(stats.get('REB', 5))
        ast = float(stats.get('AST', 4))
        stl = float(stats.get('STL', 1.0))
        blk = float(stats.get('BLK', 0.5))
        tov = float(stats.get('TOV', 2.0))
        plus_minus = float(stats.get('PLUS_MINUS', 0.0))
        fantasy_pts = float(stats.get('NBA_FANTASY_PTS', 30))
        fga = float(stats.get('FGA', 12))
        fta = float(stats.get('FTA', 3.5))

        # Real-time Preprocessing Execution
        min_safe = min_val + 1e-5
        pts_per_min = pts / min_safe
        fantasy_per_min = fantasy_pts / min_safe
        usage_proxy = (fga + 0.44 * fta + tov) / min_safe
        ts_prox = pts / (2 * (fga + 0.44 * fta + 1e-5))
        age_sq = age ** 2
        is_rookie_age = 1 if age <= 24 else 0
        veteran_prime = 1 if 26 <= age <= 31 else 0

        # Construct dataframe structured with exact expected dimensional features
        input_matrix = pd.DataFrame([{
            'AGE': age, 'AGE_SQ': age_sq, 'IS_ROOKIE_AGE': is_rookie_age, 'VETERAN_PRIME': veteran_prime,
            'GP': gp, 'W_PCT': w_pct, 'MIN': min_val, 'PTS': pts, 'REB': reb, 'AST': ast,
            'STL': stl, 'BLK': blk, 'TOV': tov, 'PLUS_MINUS': plus_minus, 'NBA_FANTASY_PTS': fantasy_pts,
            'PTS_PER_MIN': pts_per_min, 'FANTASY_PER_MIN': fantasy_per_min, 'USAGE_PROXY': usage_proxy, 'TS_PROX': ts_prox
        }])

        # Perform live execution pass
        valuation_prediction = live_ensemble_model.predict(input_matrix)[0]
        valuation_prediction = max(500000, float(valuation_prediction))

        return jsonify({'valuation': round(valuation_prediction, 2)})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    # Runs automatically on your standard local workspace setup
    app.run(debug=True, port=8000)