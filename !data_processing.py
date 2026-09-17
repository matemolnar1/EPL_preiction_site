import pandas as pd
import numpy as np

def rename_to_team_perspective(df_to_rename, perspective):
    """
    Standardizes column names to a Team vs. Opponent perspective 
    based on whether the team played at Home or Away.
    """
    new_cols = {}
    for col in df_to_rename.columns:
        if perspective == 'Home':
            if col.startswith('Home_'): 
                new_cols[col] = col.replace('Home_', 'Team_')
            elif col.startswith('Away_'): 
                new_cols[col] = col.replace('Away_', 'Opp_')
        else:
            if col.startswith('Away_'): 
                new_cols[col] = col.replace('Away_', 'Team_')
            elif col.startswith('Home_'): 
                new_cols[col] = col.replace('Home_', 'Opp_')
            
    df_to_rename.rename(columns=new_cols, inplace=True)
    df_to_rename['Is_Home'] = 1 if perspective == 'Home' else 0
    return df_to_rename

def get_kimenetel(h, a):
    """Calculates match outcome: 1 (Home Win), 0 (Draw), 2 (Away Win)."""
    if h > a: return 1
    elif h == a: return 0
    else: return 2

if __name__ == "__main__":
    print("Loading raw database...")
    df = pd.read_csv("foci_adatbazis_advanced.csv")

    df['Match_Date'] = pd.to_datetime(df['Match_Date'])
    df.sort_values(by='Match_Date', inplace=True)
    df.dropna(subset=['Match_Date'], inplace=True) 

    print("Cleaning and unpivoting timelines...")
    home_df = rename_to_team_perspective(df.copy(), 'Home')
    away_df = rename_to_team_perspective(df.copy(), 'Away')

    team_timeline = pd.concat([home_df, away_df], ignore_index=True)
    team_timeline.sort_values(by=['Team_Team', 'Match_Date'], inplace=True)

    print("Handling missing data and calculating EMA...")
    cols_missing_as_zero = [
        'Touches_OppBox', 'AccurateLongBalls', 'DistanceCovered_km', 
        'Sprints', 'LineBreakingPasses'
    ]

    for col in cols_missing_as_zero:
        team_timeline[f'Team_{col}'] = team_timeline[f'Team_{col}'].replace(0.0, np.nan)
        team_timeline[f'Opp_{col}'] = team_timeline[f'Opp_{col}'].replace(0.0, np.nan)

    base_metrics = [
        'xGOT', 'xG_OpenPlay', 'xG_SetPlay', 'xA', 'BigChancesMissed', 
        'Touches_OppBox', 'AccurateLongBalls', 'DistanceCovered_km', 
        'Sprints', 'LineBreakingPasses', 'Momentum_Avg', 'Unavailable'
    ]

    cols_to_ema = [f"Team_{m}" for m in base_metrics] + [f"Opp_{m}" for m in base_metrics]

    for col in cols_to_ema:
        team_timeline[f'{col}_EMA_5'] = team_timeline.groupby('Team_Team')[col].transform(lambda x: x.ewm(span=5, adjust=False, ignore_na=True).mean())
        team_timeline[f'{col}_EMA_20'] = team_timeline.groupby('Team_Team')[col].transform(lambda x: x.ewm(span=20, adjust=False, ignore_na=True).mean())

    print("Shifting timelines to capture pre-match state...")
    ema_cols = [col for col in team_timeline.columns if '_EMA_' in col]

    team_timeline = team_timeline.copy()

    for col in ema_cols:
        team_timeline[f'PreMatch_{col}'] = team_timeline.groupby('Team_Team')[col].shift(1)

    team_timeline.dropna(subset=[f'PreMatch_{ema_cols[0]}'], inplace=True)

    print("Rebuilding match-level matrix and applying tier filters...")
    ml_home = team_timeline[team_timeline['Is_Home'] == 1].copy()
    ml_away = team_timeline[team_timeline['Is_Home'] == 0].copy()

    home_rename = {col: col.replace('PreMatch_Team_', 'Home_Team_').replace('PreMatch_Opp_', 'Home_Opp_') for col in team_timeline.columns if 'PreMatch_' in col}
    away_rename = {col: col.replace('PreMatch_Team_', 'Away_Team_').replace('PreMatch_Opp_', 'Away_Opp_') for col in team_timeline.columns if 'PreMatch_' in col}

    ml_home.rename(columns=home_rename, inplace=True)
    ml_away.rename(columns=away_rename, inplace=True)

    final_matrix = pd.merge(
        ml_home[['Meccs_ID', 'League_Tier', 'Match_Date', 'Team_Team', 'Opp_Team', 'Team_Goals', 'Opp_Goals'] + list(home_rename.values())],
        ml_away[['Meccs_ID'] + list(away_rename.values())],
        on='Meccs_ID'
    )

    final_matrix.rename(columns={'Team_Team': 'Home_Team', 'Opp_Team': 'Away_Team', 'Team_Goals': 'Home_Goals', 'Opp_Goals': 'Away_Goals'}, inplace=True)
    final_matrix['Kimenetel'] = final_matrix.apply(lambda r: get_kimenetel(r['Home_Goals'], r['Away_Goals']), axis=1)

    pl_matrix = final_matrix[final_matrix['League_Tier'] == 1].copy()
    pl_matrix.to_csv("ml_tanito_matrix_advanced.csv", index=False)
    
    print(f"Success: Training matrix created with {len(pl_matrix)} matches.")