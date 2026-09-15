import pandas as pd
import numpy as np
import os
import glob
import warnings

warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)

def clean_name(name):
    """Standardizes team names to allow for joining model predictions with bookmaker odds."""
    if pd.isna(name): return ""
    name = str(name).lower().strip()
    if "manchester city" in name or "man city" in name: return "mancity"
    if "manchester united" in name or "man united" in name or "man utd" in name: return "manutd"
    if "brighton" in name: return "brighton"
    if "tottenham" in name or "spurs" in name: return "tottenham"
    if "west ham" in name: return "westham"
    if "wolves" in name or "wolverhampton" in name: return "wolves"
    if "nottingham" in name or "nott" in name: return "nottmforest"
    if "sheffield" in name: return "sheffutd"
    if "bournemouth" in name: return "bournemouth"
    if "leicester" in name: return "leicester"
    if "leeds" in name: return "leeds"
    if "ipswich" in name: return "ipswich"
    if "luton" in name: return "luton"
    if "newcastle" in name: return "newcastle"
    if "aston villa" in name: return "astonvilla"
    if "crystal palace" in name: return "crystalpalace"
    return name.replace(" ", "").replace("'", "")

def run_backtest(df_merged, bankroll=1000.0, edge_threshold=0.05, kelly_multiplier=0.25, min_odds=2.0, max_odds=5.0):
    """Simulates a fractional Kelly Criterion betting strategy over historical match data."""
    fogadasok_szama = 0
    nyert_fogadasok = 0
    fogadas_naplo = []

    df_merged = df_merged.sort_values(by='Match_Date')

    for index, row in df_merged.iterrows():
        legjobb_ev = max(row['EV_Hazai'], row['EV_Dontetlen'], row['EV_Vendeg'])
        
        if pd.isna(legjobb_ev) or legjobb_ev < edge_threshold:
            continue

        tipp = ""
        valoszinuseg = 0.0
        odds = 0.0
        kimenetel_id = -1
        
        if legjobb_ev == row['EV_Hazai']:
            tipp, valoszinuseg, odds, kimenetel_id = "Hazai", row['Esely_Hazai'], row['Close_H'], 1
        elif legjobb_ev == row['EV_Dontetlen']:
            tipp, valoszinuseg, odds, kimenetel_id = "Döntetlen", row['Esely_Dontetlen'], row['Close_D'], 0
        elif legjobb_ev == row['EV_Vendeg']:
            tipp, valoszinuseg, odds, kimenetel_id = "Vendég", row['Esely_Vendeg'], row['Close_A'], 2

        if odds < min_odds or odds > max_odds:
            continue

        full_kelly_pct = legjobb_ev / (odds - 1)
        bet_pct = full_kelly_pct * kelly_multiplier
        bet_pct = min(bet_pct, 0.05) 
        
        tet = bankroll * bet_pct
        fogadasok_szama += 1
        
        if row['Kimenetel'] == kimenetel_id:
            nyeremeny = (tet * odds) - tet
            bankroll += nyeremeny
            nyert_fogadasok += 1
            profit_tet = nyeremeny
        else:
            bankroll -= tet
            profit_tet = -tet
            
        fogadas_naplo.append({
            'Datum': row['Match_Date'],
            'Meccs': f"{row['Home_Team']} vs {row['Away_Team']}",
            'Megjatszott_Tipp': tipp,
            'Odds': odds,
            'Edge (%)': round(legjobb_ev * 100, 2),
            'Tet_Merete': round(tet, 2),
            'Profit_Ezen_A_Meccsen': round(profit_tet, 2),
            'Uj_Bankroll': round(bankroll, 2)
        })
        
    return bankroll, fogadasok_szama, nyert_fogadasok, fogadas_naplo


if __name__ == "__main__":
    print("Loading model predictions and probabilities...")
    try:
        df_pred = pd.read_csv("xgboost_backtest_eredmenyek.csv")
    except FileNotFoundError:
        print("Error: xgboost_backtest_eredmenyek.csv not found. Run the training script first.")
        exit(1)

    print("Locating and parsing historical odds files...")
    csv_files = glob.glob("2*.csv")
    df_odds_list = []
    
    for file in csv_files:
        temp_df = pd.read_csv(file)
        base_cols = ['Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG']
        
        if 'PSCH' in temp_df.columns:
            odds_map = {'PSCH': 'Close_H', 'PSCD': 'Close_D', 'PSCA': 'Close_A'}
        elif 'B365CH' in temp_df.columns:
            odds_map = {'B365CH': 'Close_H', 'B365CD': 'Close_D', 'B365CA': 'Close_A'}
        elif 'PSH' in temp_df.columns:
            odds_map = {'PSH': 'Close_H', 'PSD': 'Close_D', 'PSA': 'Close_A'}
        elif 'B365H' in temp_df.columns:
            odds_map = {'B365H': 'Close_H', 'B365D': 'Close_D', 'B365A': 'Close_A'}
        else:
            continue

        temp_df = temp_df.rename(columns=odds_map)
        cols_to_keep = base_cols + ['Close_H', 'Close_D', 'Close_A']
        
        if all(col in temp_df.columns for col in cols_to_keep):
            df_odds_list.append(temp_df[cols_to_keep])

    if not df_odds_list:
        print("Error: No valid historical odds files found.")
        exit(1)

    df_odds = pd.concat(df_odds_list, ignore_index=True)

    df_pred['Merge_Home'] = df_pred['Home_Team'].apply(clean_name)
    df_pred['Merge_Away'] = df_pred['Away_Team'].apply(clean_name)
    df_odds['Merge_Home'] = df_odds['HomeTeam'].apply(clean_name)
    df_odds['Merge_Away'] = df_odds['AwayTeam'].apply(clean_name)

    print("Merging predictions with historical odds...")
    df_merged = pd.merge(
        df_pred, df_odds, 
        left_on=['Merge_Home', 'Merge_Away', 'Home_Goals', 'Away_Goals'], 
        right_on=['Merge_Home', 'Merge_Away', 'FTHG', 'FTAG'], 
        how='inner'
    )
    df_merged = df_merged.drop_duplicates(subset=['Merge_Home', 'Merge_Away', 'Match_Date'])

    df_merged['EV_Hazai'] = (df_merged['Esely_Hazai'] * df_merged['Close_H']) - 1
    df_merged['EV_Dontetlen'] = (df_merged['Esely_Dontetlen'] * df_merged['Close_D']) - 1
    df_merged['EV_Vendeg'] = (df_merged['Esely_Vendeg'] * df_merged['Close_A']) - 1

    print("\nRunning Optimized Kelly Strategy Simulation...")
    STARTING_BANKROLL = 1000.0
    
    final_bankroll, total_bets, won_bets, bet_log = run_backtest(
        df_merged, 
        bankroll=STARTING_BANKROLL, 
        edge_threshold=0.05, 
        kelly_multiplier=0.25, 
        min_odds=2.0, 
        max_odds=5.0
    )

    if total_bets > 0:
        profit = final_bankroll - STARTING_BANKROLL
        roi_bankroll = (profit / STARTING_BANKROLL) * 100
        
        print(f"\n--- KELLY BACKTEST RESULTS (Odds: 2.0 - 5.0) ---")
        print(f"Starting Bankroll: {STARTING_BANKROLL:.2f} Units")
        print(f"Total Value Bets Placed: {total_bets}")
        print(f"Winning Bets: {won_bets} (Winrate: {(won_bets/total_bets)*100:.2f}%)")
        print(f"Final Bankroll: {final_bankroll:.2f} Units")
        print(f"Net Profit: {profit:.2f} Units")
        print(f"Bankroll Growth (ROI): {roi_bankroll:.2f}%")
        
        naplo_df = pd.DataFrame(bet_log)
        naplo_df.to_csv("fogadasi_naplo_optimalizalt.csv", index=False)
        print("\nDetailed betting log saved to: fogadasi_naplo_optimalizalt.csv")
    else:
        print("The model did not find any value bets within the specified odds range.")