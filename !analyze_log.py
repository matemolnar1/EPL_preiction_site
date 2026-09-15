import pandas as pd

def analyze_odds_buckets(csv_path="fogadasi_naplo_valos.csv"):
    """
    Analyzes betting performance and ROI segmented by odds categories.
    """
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"Error: Could not find {csv_path}. Please ensure the log exists.")
        return None

    if 'Valos_Odds' not in df.columns or 'Profit_Ezen_A_Meccsen' not in df.columns:
        print("Error: Required columns missing from the CSV.")
        return None

    bins = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 100.0]
    labels = [
        '1.0 - 2.0 (Favorit)', 
        '2.0 - 3.0 (Enyhe Favorit)', 
        '3.0 - 4.0 (Döntetlen/Enyhe Underdog)', 
        '4.0 - 5.0 (Underdog)', 
        '5.0 - 6.0 (Nagy Underdog)', 
        '6.0+ (Extrém Underdog)'
    ]

    df['Odds_Kategoria'] = pd.cut(df['Valos_Odds'], bins=bins, labels=labels)

    elemzes = df.groupby('Odds_Kategoria', observed=False).agg(
        Fogadasok_Szama=('Profit_Ezen_A_Meccsen', 'count'),
        Nyertes_Fogadasok=('Profit_Ezen_A_Meccsen', lambda x: (x > 0).sum()),
        Netto_Profit=('Profit_Ezen_A_Meccsen', 'sum')
    ).reset_index()

    elemzes['Winrate (%)'] = (elemzes['Nyertes_Fogadasok'] / elemzes['Fogadasok_Szama'] * 100).fillna(0)
    elemzes['ROI (%)'] = (elemzes['Netto_Profit'] / elemzes['Fogadasok_Szama'] * 100).fillna(0)

    return elemzes

if __name__ == "__main__":
    print("Loading betting log and calculating bucket metrics...")
    analysis_results = analyze_odds_buckets()

    if analysis_results is not None:
        print("\n--- BETTING LOG ANALYSIS BY ODDS BUCKET ---")
        for index, row in analysis_results.iterrows():
            if row['Fogadasok_Szama'] > 0:
                print(f"\nBucket: {row['Odds_Kategoria']}")
                print(f" Bets: {row['Fogadasok_Szama']} | Won: {row['Nyertes_Fogadasok']} ({row['Winrate (%)']:.1f}%)")
                print(f" Net Profit: {row['Netto_Profit']:.2f} Units | ROI: {row['ROI (%)']:.2f}%")