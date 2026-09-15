import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import accuracy_score, classification_report
import warnings
import joblib

warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)

def engineer_features(df):
    """Calculates performance differentials and finishing skill metrics, returning the clean dataframe."""
    metrics_to_diff = [
        'xG_OpenPlay_EMA_20', 
        'Touches_OppBox_EMA_20', 
        'xGOT_EMA_20', 
        'Sprints_EMA_5', 
        'BigChancesMissed_EMA_20'
    ]

    for m in metrics_to_diff:
        df[f'{m}_Att_Diff'] = df[f'Home_Team_{m}'] - df[f'Away_Team_{m}']
        df[f'{m}_Def_Diff'] = df[f'Home_Opp_{m}'] - df[f'Away_Opp_{m}']

    df['Home_Finishing_Skill_20'] = df['Home_Team_xGOT_EMA_20'] - df['Home_Team_xG_OpenPlay_EMA_20']
    df['Away_Finishing_Skill_20'] = df['Away_Team_xGOT_EMA_20'] - df['Away_Team_xG_OpenPlay_EMA_20']
    df['Finishing_Diff'] = df['Home_Finishing_Skill_20'] - df['Away_Finishing_Skill_20']

    raw_cols_to_drop = []
    for m in metrics_to_diff:
        raw_cols_to_drop.extend([f'Home_Team_{m}', f'Away_Team_{m}', f'Home_Opp_{m}', f'Away_Opp_{m}'])

    cols_to_drop = [
        'Meccs_ID', 'League_Tier', 'Match_Date', 
        'Home_Team', 'Away_Team', 'Home_Goals', 'Away_Goals', 'Kimenetel',
        'Home_PreMatch_Elo', 'Away_PreMatch_Elo', 'Home_Rest_Days', 'Away_Rest_Days',
        'Home_Finishing_Skill_20', 'Away_Finishing_Skill_20'
    ] + raw_cols_to_drop
    
    return df, cols_to_drop

if __name__ == "__main__":
    print("Loading training matrix and applying feature engineering...")
    df = pd.read_csv("ml_tanito_matrix_vegsleg.csv")
    df['Match_Date'] = pd.to_datetime(df['Match_Date'])
    df.sort_values(by='Match_Date', inplace=True)
    df = df.copy()

    df, cols_to_drop = engineer_features(df)

    X = df.drop(columns=[c for c in cols_to_drop if c in df.columns])
    y = df['Kimenetel'] 

    test_size = 380
    split_index = len(df) - test_size
    X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
    y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]

    print(f"\n[1/2] Evaluating model on validation set...")
    print(f"Training set: {len(X_train)} matches | Testing set: {len(X_test)} matches")
    print(f"Number of features used: {len(X.columns)}")

    xgb_params = {
        'objective': 'multi:softprob',
        'num_class': 3,
        'max_depth': 3,                
        'learning_rate': 0.01,         
        'n_estimators': 1000,          
        'subsample': 0.8,
        'colsample_bytree': 0.7,
        'gamma': 1.0,                  
        'random_state': 42
    }

    xgb_eval = xgb.XGBClassifier(**xgb_params)
    xgb_eval.fit(X_train, y_train)

    y_pred = xgb_eval.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nTest Set Accuracy: {accuracy * 100:.2f}%")
    print(classification_report(y_test, y_pred, target_names=['Draw (0)', 'Home (1)', 'Away (2)'], zero_division=0))

    y_proba = xgb_eval.predict_proba(X_test)
    eredmenyek = df.iloc[split_index:].copy()
    eredmenyek = eredmenyek[['Match_Date', 'Home_Team', 'Away_Team', 'Home_Goals', 'Away_Goals', 'Kimenetel']]
    eredmenyek['Modell_Tipp'] = y_pred
    eredmenyek['Esely_Dontetlen'] = y_proba[:, 0]
    eredmenyek['Esely_Hazai'] = y_proba[:, 1]
    eredmenyek['Esely_Vendeg'] = y_proba[:, 2]
    eredmenyek.to_csv("xgboost_backtest_eredmenyek.csv", index=False)
    print("-> Backtest results saved to: xgboost_backtest_eredmenyek.csv")

    print(f"\n[2/2] Training production model on the full dataset ({len(X)} matches)...")
    
    xgb_production = xgb.XGBClassifier(**xgb_params)
    xgb_production.fit(X, y)

    joblib.dump(xgb_production, 'xgboost_epl_model.pkl')
    print("Success! Production model trained and saved to: xgboost_epl_model.pkl")