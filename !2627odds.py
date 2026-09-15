import pandas as pd

def download_historical_odds(season_code, league_code="E0"):
    """
    Downloads historical match odds from football-data.co.uk.
    
    Parameters:
    season_code (str): The season format (e.g., '2627' for 2026/2027).
    league_code (str): The league format (e.g., 'E0' for Premier League).
    """
    url = f"https://www.football-data.co.uk/mmz4281/{season_code}/{league_code}.csv"
    
    try:
        print(f"Attempting to download data from: {url}")
        df = pd.read_csv(url)
        
        output_filename = f"{season_code}.csv"
        df.to_csv(output_filename, index=False)
        print(f"Success! Data saved locally as: {output_filename}")
        
    except Exception as e:
        print(f"Error downloading the data: {e}")

if __name__ == "__main__":
    # Example usage:
    download_historical_odds("2627", "E0")