import requests
import pandas as pd
import time
import os
import random
import numpy as np

def get_stat(stats_list, target_key):
    """Safely extracts home and away statistical values from the parsed JSON."""
    if not stats_list:
        return [0, 0]
    
    for category in stats_list:
        for stat in category.get('stats', []):
            if stat.get('key') == target_key:
                try:
                    val_h = float(str(stat.get('stats', [0, 0])[0]).replace('%', ''))
                    val_a = float(str(stat.get('stats', [0, 0])[1]).replace('%', ''))
                    return [val_h, val_a]
                except (ValueError, TypeError):
                    return [0, 0]
    return [0, 0]

def extract_match_features(m_id, headers, tier):
    """Fetches match details from FotMob API and parses relevant advanced metrics."""
    url = f"https://www.fotmob.com/api/data/matchDetails?matchId={m_id}"
    res = requests.get(url, headers=headers)
    
    if res.status_code != 200:
        return None
    
    data = res.json()
    content = data.get('content') or {}
    general_data = data.get('general') or {}
    
    match_info = {
        'Meccs_ID': int(m_id),
        'Match_Date': general_data.get('matchTimeUTCDate'), 
        'Season': general_data.get('parentLeagueSeason', 'Unknown'),
        'League_ID': general_data.get('leagueId'),
        'League_Tier': tier,
        'Home_Team': (general_data.get('homeTeam') or {}).get('name', 'Unknown'),
        'Away_Team': (general_data.get('awayTeam') or {}).get('name', 'Unknown'),
    }
    
    header_data = data.get('header') or {}
    teams_header = header_data.get('teams') or []
    if len(teams_header) >= 2:
        match_info['Home_Goals'] = (teams_header[0] or {}).get('score', 0)
        match_info['Away_Goals'] = (teams_header[1] or {}).get('score', 0)
    else:
        match_info['Home_Goals'] = 0
        match_info['Away_Goals'] = 0
    
    lineup = content.get('lineup') or {}
    home_lineup = lineup.get('homeTeam') or {}
    away_lineup = lineup.get('awayTeam') or {}
    
    match_info['Home_Unavailable'] = len(home_lineup.get('unavailable') or [])
    match_info['Away_Unavailable'] = len(away_lineup.get('unavailable') or [])
    
    stats_data = content.get('stats') or {}
    periods_data = stats_data.get('Periods') or {}
    all_periods = periods_data.get('All') or {}
    stats_all = all_periods.get('stats') or []
    
    keys_to_extract = [
        ('expected_goals_on_target', 'xGOT'),
        ('expected_goals_open_play', 'xG_OpenPlay'),
        ('expected_goals_set_play', 'xG_SetPlay'),
        ('expected_assists', 'xA'),
        ('big_chance_missed_title', 'BigChancesMissed'),
        ('touches_opp_box', 'Touches_OppBox'),
        ('long_balls_accurate', 'AccurateLongBalls'),
        ('physical_metrics_distance_covered', 'DistanceCovered_km'),
        ('physical_metrics_number_of_sprints', 'Sprints')
    ]
    
    for fotmob_key, my_col in keys_to_extract:
        h_val, a_val = get_stat(stats_all, fotmob_key)
        match_info[f'Home_{my_col}'] = h_val
        match_info[f'Away_{my_col}'] = a_val

    player_stats_dict = content.get('playerStats') or {}
    home_lbp = 0
    away_lbp = 0
    home_team_id = general_data.get('homeTeam', {}).get('id')
    
    for player_id, p_data in player_stats_dict.items():
        if not p_data: continue
        p_team_id = p_data.get('teamId')
        p_stats_array = p_data.get('stats') or []
        
        lb_passes = 0
        for stat_group in p_stats_array:
            if not stat_group: continue
            stats_vals = stat_group.get('stats') or {}
            for s in stats_vals.values():
                if s and s.get('key') == 'line_breaking_passes':
                    lb_passes = s.get('value', 0)
                    break
                    
        if p_team_id == home_team_id:
            home_lbp += lb_passes
        else:
            away_lbp += lb_passes
            
    match_info['Home_LineBreakingPasses'] = home_lbp
    match_info['Away_LineBreakingPasses'] = away_lbp

    match_facts = content.get('matchFacts') or {}
    momentum = match_facts.get('momentum') or {}
    main_mom = momentum.get('main') or {}
    momentum_data = main_mom.get('data') or []
    
    if momentum_data:
        values = [point.get('value', 0) for point in momentum_data if point]
        match_info['Home_Momentum_Avg'] = np.mean([v for v in values if v > 0]) if any(v > 0 for v in values) else 0
        match_info['Away_Momentum_Avg'] = np.mean([abs(v) for v in values if v < 0]) if any(v < 0 for v in values) else 0
    else:
        match_info['Home_Momentum_Avg'] = 0
        match_info['Away_Momentum_Avg'] = 0

    return match_info


if __name__ == "__main__":
    leagues_to_scrape = {
        47: 1, 
        48: 2, 
        108: 3 
    }

    szezonok = [
        "2021/2022", "2022/2023", "2023/2024", 
        "2024/2025", "2025/2026", "2026/2027"
    ]

    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Accept-Language': 'en-US,en;q=0.9',
        'Accept': 'application/json'
    }

    CSV_FAJL = "foci_adatbazis_advanced.csv"

    for league_id, league_tier in leagues_to_scrape.items():
        print(f"\nProcessing Tier {league_tier} (ID: {league_id})...")
        
        lejatszott_id_lista = []

        for szezon in szezonok:
            szezon_url = szezon.replace("/", "%2F")
            url_league = f"https://www.fotmob.com/api/data/leagues?id={league_id}&season={szezon_url}&ccode3=HUN"
            
            res = requests.get(url_league, headers=headers)
            if res.status_code == 200:
                matches = res.json().get('fixtures', {}).get('allMatches', [])
                for match in matches:
                    if match.get('status', {}).get('finished') == True:
                        lejatszott_id_lista.append(int(match.get('id')))
            else:
                print(f"-> Error fetching season {szezon} (Status: {res.status_code})")
            
            time.sleep(1)

        lejatszott_id_lista = list(set(lejatszott_id_lista)) 

        meglevo_meccs_idk = []
        if os.path.exists(CSV_FAJL):
            df_regi = pd.read_csv(CSV_FAJL)
            meglevo_meccs_idk = df_regi['Meccs_ID'].unique().tolist()

        letoltendo_idk = [m_id for m_id in lejatszott_id_lista if m_id not in meglevo_meccs_idk]
        
        if not letoltendo_idk:
            continue

        print(f"Downloading {len(letoltendo_idk)} new matches for Tier {league_tier}...")

        kibovitett_meccsek = []
        mentes_gyakorisag = 20 

        for i, m_id in enumerate(letoltendo_idk):
            time.sleep(random.uniform(1, 2)) 
            
            meccs_adat = extract_match_features(str(m_id), headers, league_tier)
            
            if meccs_adat:
                kibovitett_meccsek.append(meccs_adat)
            else:
                print(f"-> Error downloading ID {m_id}. (Possible Rate-limit)")
                time.sleep(10) 
                
            if (i + 1) % mentes_gyakorisag == 0 or (i + 1) == len(letoltendo_idk):
                if kibovitett_meccsek:
                    df_uj = pd.DataFrame(kibovitett_meccsek)
                    
                    if os.path.exists(CSV_FAJL):
                        df_regi = pd.read_csv(CSV_FAJL, nrows=0) 
                        df_uj = df_uj.reindex(columns=df_regi.columns)
                        df_uj.to_csv(CSV_FAJL, mode='a', header=False, index=False)
                    else:
                        df_uj.to_csv(CSV_FAJL, mode='w', header=True, index=False)
                        
                    print(f"*** Checkpoint: {len(kibovitett_meccsek)} matches saved to {CSV_FAJL}. ***")
                    kibovitett_meccsek = [] 

    print("\nDatabase update complete.")