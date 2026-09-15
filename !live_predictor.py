import os
import json
import time
import requests
import joblib
import warnings
import pandas as pd
from datetime import datetime, timezone, timedelta

warnings.filterwarnings('ignore', category=pd.errors.PerformanceWarning)

def clean_name(name):
    """Standardizes team names for reliable database merging."""
    if pd.isna(name): return ""
    name = str(name).lower().strip()
    
    if "hull" in name: return "hull"
    if "coventry" in name: return "coventry"
    if "sunderland" in name: return "sunderland"
    if "man" in name and "city" in name: return "mancity"
    if "man" in name and ("utd" in name or "united" in name): return "manutd"
    if "tottenham" in name or "spurs" in name: return "tottenham"
    if "nott" in name or "forest" in name: return "nottmforest"
    if "wolves" in name or "wolverhampton" in name: return "wolves"
    if "sheff" in name: return "sheffutd"
    
    replacements = {
        "west ham": "westham", "bournemouth": "bournemouth",
        "leicester": "leicester", "leeds": "leeds", "ipswich": "ipswich",
        "luton": "luton", "newcastle": "newcastle", "aston villa": "astonvilla",
        "crystal palace": "crystalpalace", "brighton": "brighton",
        "brentford": "brentford", "everton": "everton", "fulham": "fulham",
        "liverpool": "liverpool", "arsenal": "arsenal", "chelsea": "chelsea",
        "southampton": "southampton"
    }
    for k, v in replacements.items():
        if k in name: return v
        
    return name.replace(" ", "").replace("'", "")

def parse_to_budapest_time(time_str):
    """Converts UTC match times to Budapest local time."""
    if not time_str:
        return "-", 99
    try:
        cleaned = str(time_str).replace("Z", "+00:00")
        dt_utc = datetime.fromisoformat(cleaned)
        if dt_utc.tzinfo is None:
            dt_utc = dt_utc.replace(tzinfo=timezone.utc)
            
        try:
            from zoneinfo import ZoneInfo
            dt_bp = dt_utc.astimezone(ZoneInfo("Europe/Budapest"))
        except Exception:
            m = dt_utc.month
            offset = 2 if (4 <= m <= 10) else 1  
            dt_bp = dt_utc.astimezone(timezone(timedelta(hours=offset)))
            
        now_bp = datetime.now(dt_bp.tzinfo)
        days_diff = (dt_bp.date() - now_bp.date()).days
        return dt_bp.strftime("%Y-%m-%d %H:%M"), days_diff
    except Exception:
        return str(time_str)[:16], 99

def get_match_odds(match_id, headers):
    """Fetches real-time match odds from the FotMob API."""
    url = f"https://www.fotmob.com/api/data/matchOdds?matchId={match_id}&ccode3=HUN"
    try:
        r = requests.get(url, headers=headers, timeout=5).json()
        markets = r.get('odds', {}).get('matchfactMarkets', [])
        for m in markets:
            if m.get('header') == "Match Odds":
                h, d, a = 0.0, 0.0, 0.0
                for sel in m.get('selections', []):
                    name = str(sel.get('name', '')).lower()
                    val = float(sel.get('oddsDecimal') or 0.0)
                    if name == "1": h = val
                    elif name == "x": d = val
                    elif name == "2": a = val
                return h, d, a
    except Exception: 
        pass
    return 0.0, 0.0, 0.0

def extract_matches(obj):
    """Recursively finds upcoming match objects from the API JSON response."""
    found = []
    if isinstance(obj, dict):
        if ('home' in obj or 'homeTeam' in obj) and ('away' in obj or 'awayTeam' in obj) and 'id' in obj:
            found.append(obj)
        for v in obj.values(): found.extend(extract_matches(v))
    elif isinstance(obj, list):
        for item in obj: found.extend(extract_matches(item))
    return found

def evaluate_outcome(p, odds):
    """Calculates Expected Value (EV) and Fractional Kelly Criterion."""
    if odds <= 0: return 0.0, 0.0, "NINCS ODDS"
    ev = (p * odds) - 1.0
    ev_pct = round(float(ev) * 100, 2)
    if ev <= 0: return ev_pct, 0.0, "⚪ NINCS ÉRTÉK"
    
    full_k = ev / (odds - 1.0)
    kelly_quarter = round(float(full_k / 4.0) * 100, 2)
    
    if ev_pct > 30.0:
        return ev_pct, 0.0, "⚠️ ANOMÁLIA"
    elif odds < 2.0 or odds > 5.0:
        return ev_pct, 0.0, "🟡 ODDS SZŰRVE"
    else:
        return ev_pct, min(kelly_quarter, 5.0), "🔥 VALUE BET"


if __name__ == "__main__":
    print("Loading model and training matrix...")
    
    model_file = "xgboost_epl_model.pkl" if os.path.exists("xgboost_epl_model.pkl") else "ml_foci_agy.pkl"
    if not os.path.exists(model_file):
        print(f"Error: Trained model not found ({model_file})!")
        exit(1)
    
    model = joblib.load(model_file)
    
    matrix_file = "ml_tanito_matrix_vegsleg.csv"
    if not os.path.exists(matrix_file):
        print(f"Error: Matrix file '{matrix_file}' not found!")
        exit(1)
    
    df = pd.read_csv(matrix_file)
    df['Match_Date'] = pd.to_datetime(df['Match_Date'])
    df.sort_values(by='Match_Date', inplace=True)
    df = df.copy()

    df['Clean_Home'] = df['Home_Team'].apply(clean_name)
    df['Clean_Away'] = df['Away_Team'].apply(clean_name)
    ismert_csapatok = set(df['Clean_Home'].unique()).union(set(df['Clean_Away'].unique()))

    print("Fetching FotMob data...")
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Accept': 'application/json'
    }

    url_league = "https://www.fotmob.com/api/data/leagues?id=47&ccode3=HUN"
    try:
        res = requests.get(url_league, headers=headers, timeout=10).json()
    except Exception as e:
        print(f"Error accessing FotMob API: {e}")
        exit(1)

    try:
        table_rows = []
        table_data = res.get('table', [])
        if isinstance(table_data, list) and len(table_data) > 0:
            raw_table = table_data[0].get('data', {}).get('table', {}).get('all', [])
            for row in raw_table:
                t_id = row.get('id')
                t_name = row.get('name')
                c_name = clean_name(t_name)
                
                t_elo = 1500.0
                t_xg_att = 0.0
                t_xg_def = 0.0
                last_5 = []
                
                t_matches = df[(df['Clean_Home'] == c_name) | (df['Clean_Away'] == c_name)]
                if len(t_matches) > 0:
                    last_row = t_matches.iloc[-1]
                    is_h = (last_row['Clean_Home'] == c_name)
                    t_elo = round(float(last_row['Home_PreMatch_Elo' if is_h else 'Away_PreMatch_Elo']), 1)
                    prefix = 'Home_Team_' if is_h else 'Away_Team_'
                    opp_prefix = 'Home_Opp_' if is_h else 'Away_Opp_'
                    t_xg_att = round(float(last_row.get(prefix + 'xG_OpenPlay_EMA_20', 0)), 2)
                    t_xg_def = round(float(last_row.get(opp_prefix + 'xG_OpenPlay_EMA_20', 0)), 2)
                    
                    for _, m_row in t_matches.tail(5).iterrows():
                        m_is_h = (m_row['Clean_Home'] == c_name)
                        opp = m_row['Away_Team'] if m_is_h else m_row['Home_Team']
                        hg = int(m_row['Home_Goals']) if pd.notna(m_row['Home_Goals']) else 0
                        ag = int(m_row['Away_Goals']) if pd.notna(m_row['Away_Goals']) else 0
                        
                        if hg == ag: res_char = 'D'
                        elif (m_is_h and hg > ag) or (not m_is_h and ag > hg): res_char = 'W'
                        else: res_char = 'L'
                        
                        last_5.append({
                            'res': res_char,
                            'score': f"{hg}-{ag}",
                            'opp': opp,
                            'is_home': m_is_h
                        })

                table_rows.append({
                    'rank': int(row.get('idx', 0)),
                    'id': int(t_id) if t_id else 0,
                    'team': str(t_name),
                    'logo': f"https://images.fotmob.com/image_resources/logo/teamlogo/{t_id}.png",
                    'played': int(row.get('played', 0)),
                    'wins': int(row.get('wins', 0)),
                    'draws': int(row.get('draws', 0)),
                    'losses': int(row.get('losses', 0)),
                    'scoresStr': str(row.get('scoresStr', f"{row.get('goalsFor', 0)}-{row.get('goalsAgainst', 0)}")),
                    'goalConDiff': int(row.get('goalConDiff', 0)),
                    'pts': int(row.get('pts', 0)),
                    'elo': float(t_elo),
                    'xg_att': float(t_xg_att),
                    'xg_def': float(t_xg_def),
                    'last_5': last_5
                })

        if table_rows:
            with open("pl_table.json", "w", encoding="utf-8") as f:
                json.dump(table_rows, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"Warning: Failed to extract league table: {e}")

    raw_matches = extract_matches(res)
    kovetkezo_meccsek = []
    seen_ids = set()

    for m in raw_matches:
        mid = m.get('id')
        if not mid or mid in seen_ids: continue
        
        finished = m.get('status', {}).get('finished', False) or m.get('status', {}).get('cancelled', False)
        if finished: continue

        h_obj = m.get('home') if isinstance(m.get('home'), dict) else m.get('homeTeam', {})
        a_obj = m.get('away') if isinstance(m.get('away'), dict) else m.get('awayTeam', {})
        h_name = h_obj.get('name')
        a_name = a_obj.get('name')
        h_id = h_obj.get('id')
        a_id = a_obj.get('id')
        
        raw_time = m.get('status', {}).get('utcTime') or m.get('utcTime') or m.get('time') or ""
        readable_date, days_until = parse_to_budapest_time(raw_time)

        if h_name and a_name:
            ch, ca = clean_name(h_name), clean_name(a_name)
            if ch in ismert_csapatok and ca in ismert_csapatok:
                seen_ids.add(mid)
                kovetkezo_meccsek.append({
                    'mid': mid, 'home': h_name, 'away': a_name, 
                    'home_id': h_id, 'away_id': a_id,
                    'ch': ch, 'ca': ca, 'date': readable_date,
                    'days_until': days_until
                })
                if len(kovetkezo_meccsek) >= 25:
                    break

    print(f" -> {len(kovetkezo_meccsek)} matches scheduled...")

    tippek_naplo = []

    for m in kovetkezo_meccsek:
        mid = m['mid']
        hazai, vendeg = m['home'], m['away']
        h_id, a_id = m['home_id'], m['away_id']
        ch, ca, datum, days_until = m['ch'], m['ca'], m['date'], m['days_until']
        
        try:
            hazai_utolso = df[(df['Clean_Home'] == ch) | (df['Clean_Away'] == ch)].iloc[-1]
            vendeg_utolso = df[(df['Clean_Home'] == ca) | (df['Clean_Away'] == ca)].iloc[-1]
        except IndexError:
            continue

        def get_stats(row, cname):
            is_h = (row['Clean_Home'] == cname)
            p = 'Home_' if is_h else 'Away_'
            s = {}
            for col in row.index:
                if col.startswith(p + 'Team_'): s[f'Team_{col.replace(p + "Team_", "")}'] = row[col]
                elif col.startswith(p + 'Opp_'): s[f'Opp_{col.replace(p + "Opp_", "")}'] = row[col]
            s['Elo'] = row[p + 'PreMatch_Elo']
            return s

        hs = get_stats(hazai_utolso, ch)
        vs = get_stats(vendeg_utolso, ca)

        uj = pd.DataFrame(index=[0])
        uj['Elo_Diff'] = hs['Elo'] - vs['Elo']
        diff_metrics = ['xG_OpenPlay_EMA_20', 'Touches_OppBox_EMA_20', 'xGOT_EMA_20', 'Sprints_EMA_5', 'BigChancesMissed_EMA_20']
        for met in diff_metrics:
            uj[f'{met}_Att_Diff'] = hs.get(f'Team_{met}', 0) - vs.get(f'Team_{met}', 0)
            uj[f'{met}_Def_Diff'] = hs.get(f'Opp_{met}', 0) - vs.get(f'Opp_{met}', 0)

        uj['Finishing_Diff'] = (hs.get('Team_xGOT_EMA_20', 0) - hs.get('Team_xG_OpenPlay_EMA_20', 0)) - \
                               (vs.get('Team_xGOT_EMA_20', 0) - vs.get('Team_xG_OpenPlay_EMA_20', 0))

        for feat in model.feature_names_in_:
            if feat not in uj.columns:
                if feat.startswith('Home_Team_'): uj[feat] = hs.get(feat.replace('Home_', ''), 0)
                elif feat.startswith('Home_Opp_'): uj[feat] = hs.get(feat.replace('Home_', ''), 0)
                elif feat.startswith('Away_Team_'): uj[feat] = vs.get(feat.replace('Away_', ''), 0)
                elif feat.startswith('Away_Opp_'): uj[feat] = vs.get(feat.replace('Away_', ''), 0)
                else: uj[feat] = 0

        uj = uj[model.feature_names_in_]
        proba = model.predict_proba(uj)[0]
        
        p_draw = float(proba[0])
        p_home = float(proba[1])
        p_away = float(proba[2])

        odds_h, odds_d, odds_a = get_match_odds(mid, headers)
        time.sleep(0.3)

        fair_h = round(1.0 / p_home, 2) if p_home > 0 else 0.0
        fair_d = round(1.0 / p_draw, 2) if p_draw > 0 else 0.0
        fair_a = round(1.0 / p_away, 2) if p_away > 0 else 0.0

        ev_h, kelly_h, stat_h = evaluate_outcome(p_home, odds_h)
        ev_d, kelly_d, stat_d = evaluate_outcome(p_draw, odds_d)
        ev_a, kelly_a, stat_a = evaluate_outcome(p_away, odds_a)

        outcomes = [
            ("Hazai (1)", ev_h, kelly_h, stat_h),
            ("Döntetlen (X)", ev_d, kelly_d, stat_d),
            ("Vendég (2)", ev_a, kelly_a, stat_a)
        ]
        outcomes.sort(key=lambda x: x[1], reverse=True)
        best_tipp, best_ev, best_kelly, best_stat = outcomes[0]

        tippek_naplo.append({
            'Meccs_ID': int(mid),
            'Datum': str(datum),
            'Days_Until': int(days_until),
            'Meccs': f"{hazai} vs {vendeg}",
            'Hazai': str(hazai),
            'Vendeg': str(vendeg),
            'Home_Logo': f"https://images.fotmob.com/image_resources/logo/teamlogo/{h_id}.png" if h_id else "",
            'Away_Logo': f"https://images.fotmob.com/image_resources/logo/teamlogo/{a_id}.png" if a_id else "",
            'Hazai_%': round(float(p_home * 100), 1),
            'X_%': round(float(p_draw * 100), 1),
            'Vendeg_%': round(float(p_away * 100), 1),
            'Tipp': str(best_tipp),
            'EV_%': float(best_ev),
            'Kelly_Tet_%': float(best_kelly),
            'Statusz': str(best_stat),
            'Fair_1': float(fair_h), 'Odds_1': float(odds_h), 'EV_1': float(ev_h), 'Kelly_1': float(kelly_h), 'Status_1': str(stat_h),
            'Fair_X': float(fair_d), 'Odds_X': float(odds_d), 'EV_X': float(ev_d), 'Kelly_X': float(kelly_d), 'Status_X': str(stat_d),
            'Fair_2': float(fair_a), 'Odds_2': float(odds_a), 'EV_2': float(ev_a), 'Kelly_2': float(kelly_a), 'Status_2': str(stat_a),
            'Home_Elo': round(float(hs['Elo']), 1),
            'Away_Elo': round(float(vs['Elo']), 1),
            'Elo_Diff': round(float(hs['Elo'] - vs['Elo']), 1),
            'Home_xG_Att': round(float(hs.get('Team_xG_OpenPlay_EMA_20', 0)), 2),
            'Away_xG_Att': round(float(vs.get('Team_xG_OpenPlay_EMA_20', 0)), 2),
            'Home_xG_Def': round(float(hs.get('Opp_xG_OpenPlay_EMA_20', 0)), 2),
            'Away_xG_Def': round(float(vs.get('Opp_xG_OpenPlay_EMA_20', 0)), 2),
            'Home_Touches': round(float(hs.get('Team_Touches_OppBox_EMA_20', 0)), 1),
            'Away_Touches': round(float(vs.get('Team_Touches_OppBox_EMA_20', 0)), 1)
        })

    df_res = pd.DataFrame(tippek_naplo)
    df_res.to_csv("hetvegi_tippek.csv", index=False, encoding='utf-8-sig')

    json_records = json.loads(df_res.to_json(orient='records'))
    with open("hetvegi_tippek.json", "w", encoding="utf-8") as f:
        json.dump(json_records, f, ensure_ascii=False, indent=2)

    print(f"Done! {len(tippek_naplo)} matches processed successfully.")