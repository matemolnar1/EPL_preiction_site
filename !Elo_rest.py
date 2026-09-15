import pandas as pd
import numpy as np

# --- STRATEGY CONSTANTS ---
DEFAULT_ELO = 1500
K_FACTOR = 20
HOME_ADVANTAGE = 75 
MAX_REST_DAYS = 14 

def calculate_elo_and_rest(df):
    """
    Iterates through the match chronology to calculate dynamic Elo ratings 
    (accounting for home advantage) and team rest days.
    """
    elo_dict = {}
    last_match_date = {}

    home_elos = []
    away_elos = []
    home_rest = []
    away_rest = []

    for index, row in df.iterrows():
        home_team = row['Home_Team']
        away_team = row['Away_Team']
        match_date = row['Match_Date']
        h_goals = row['Home_Goals']
        a_goals = row['Away_Goals']

        if home_team in last_match_date:
            h_rest = (match_date - last_match_date[home_team]).days
        else:
            h_rest = MAX_REST_DAYS 

        if away_team in last_match_date:
            a_rest = (match_date - last_match_date[away_team]).days
        else:
            a_rest = MAX_REST_DAYS

        h_rest = min(h_rest, MAX_REST_DAYS)
        a_rest = min(a_rest, MAX_REST_DAYS)

        home_rest.append(h_rest)
        away_rest.append(a_rest)
        
        last_match_date[home_team] = match_date
        last_match_date[away_team] = match_date

        h_elo = elo_dict.get(home_team, DEFAULT_ELO)
        a_elo = elo_dict.get(away_team, DEFAULT_ELO)

        home_elos.append(h_elo)
        away_elos.append(a_elo)

        h_expected = 1 / (1 + 10 ** ((a_elo - (h_elo + HOME_ADVANTAGE)) / 400))
        a_expected = 1 / (1 + 10 ** (((h_elo + HOME_ADVANTAGE) - a_elo) / 400))

        if h_goals > a_goals:
            h_actual, a_actual = 1.0, 0.0
        elif h_goals == a_goals:
            h_actual, a_actual = 0.5, 0.5
        else:
            h_actual, a_actual = 0.0, 1.0

        elo_dict[home_team] = h_elo + K_FACTOR * (h_actual - h_expected)
        elo_dict[away_team] = a_elo + K_FACTOR * (a_actual - a_expected)

    df['Home_PreMatch_Elo'] = home_elos
    df['Away_PreMatch_Elo'] = away_elos
    df['Elo_Diff'] = df['Home_PreMatch_Elo'] - df['Away_PreMatch_Elo'] 

    df['Home_Rest_Days'] = home_rest
    df['Away_Rest_Days'] = away_rest
    df['Rest_Diff'] = df['Home_Rest_Days'] - df['Away_Rest_Days'] 
    
    return df


if __name__ == "__main__":
    print("Loading database and sorting chronologically...")
    df = pd.read_csv("ml_tanito_matrix_advanced.csv")

    df['Match_Date'] = pd.to_datetime(df['Match_Date'])
    df.sort_values(by='Match_Date', inplace=True)

    print("Calculating Elo Ratings and Rest Days match-by-match...")
    df = calculate_elo_and_rest(df)

    print("Appending new variables to the training matrix...")
    output_file = "ml_tanito_matrix_vegsleg.csv"
    df.to_csv(output_file, index=False)
    
    print(f"Success! Final data saved to: {output_file}")
    print("Database expanded with Elo_Diff and Rest_Diff dimensions.")