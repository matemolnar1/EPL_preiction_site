import os
import sys
import json
import sqlite3
import requests
import subprocess
import datetime
import pandas as pd
from flask import Flask, render_template, jsonify, request

app = Flask(__name__)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "bankroll.db")

def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS settings (
            key TEXT PRIMARY KEY,
            value TEXT
        )
    """)
    cursor.execute("INSERT OR IGNORE INTO settings (key, value) VALUES ('initial_bankroll', '100000')")
    cursor.execute("INSERT OR IGNORE INTO settings (key, value) VALUES ('currency', 'Ft')")
    
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS bets (
            id INTEGER PRIMARY KEY AUTOINCREMENT
        )
    """)
    
    cursor.execute("PRAGMA table_info(bets)")
    existing_columns = [col[1] for col in cursor.fetchall()]
    
    required_columns = {
        'placed_at': "TIMESTAMP DEFAULT CURRENT_TIMESTAMP",
        'match_date': "TEXT",
        'match_id': "INTEGER",
        'match_name': "TEXT",
        'selection': "TEXT",
        'odds_placed': "REAL",
        'odds_closing': "REAL",
        'stake_amount': "REAL",
        'stake_pct': "REAL",
        'ev_pct': "REAL",
        'status': "TEXT DEFAULT 'PENDING'",
        'pnl': "REAL DEFAULT 0.0",
        'settled_at': "TIMESTAMP"
    }
    
    for col_name, col_def in required_columns.items():
        if col_name not in existing_columns:
            try:
                cursor.execute(f"ALTER TABLE bets ADD COLUMN {col_name} {col_def}")
            except Exception:
                pass
                
    conn.commit()
    conn.close()

init_db()

def get_db_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def run_script(script_name):
    script_path = os.path.join(BASE_DIR, script_name)
    if not os.path.exists(script_path):
        return False, f"Script not found: {script_name}"
    
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    env["PYTHONWARNINGS"] = "ignore"

    try:
        result = subprocess.run(
            [sys.executable, script_path],
            capture_output=True,
            text=True,
            encoding='utf-8',
            errors='replace',
            cwd=BASE_DIR,
            env=env
        )
        output = result.stdout or ""
        if result.returncode != 0 and result.stderr:
            output += "\n--- ERROR (STDERR) ---\n" + result.stderr
        return (result.returncode == 0), output
    except Exception as e:
        return False, f"Execution error: {e}"

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/tips', methods=['GET'])
def get_tips():
    json_path = os.path.join(BASE_DIR, "hetvegi_tippek.json")
    csv_path = os.path.join(BASE_DIR, "hetvegi_tippek.csv")
    
    if os.path.exists(json_path):
        try:
            mtime = os.path.getmtime(json_path)
            last_mod = datetime.datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M:%S")
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return jsonify({"success": True, "data": data, "last_modified": last_mod, "count": len(data)})
        except Exception:
            pass

    if os.path.exists(csv_path):
        try:
            mtime = os.path.getmtime(csv_path)
            last_mod = datetime.datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M:%S")
            df = pd.read_csv(csv_path, encoding='utf-8-sig').fillna("")
            return jsonify({"success": True, "data": df.to_dict(orient='records'), "last_modified": last_mod, "count": len(df)})
        except Exception as e:
            return jsonify({"success": False, "message": str(e)})

    return jsonify({"success": False, "message": "No saved data available yet."})

@app.route('/api/table', methods=['GET'])
def get_table():
    table_path = os.path.join(BASE_DIR, "pl_table.json")
    if os.path.exists(table_path):
        try:
            with open(table_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return jsonify({"success": True, "table": data})
        except Exception as e:
            return jsonify({"success": False, "message": str(e)})
    return jsonify({"success": False, "message": "No saved table available yet."})

@app.route('/api/bankroll/summary', methods=['GET'])
def get_bankroll_summary():
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute("SELECT value FROM settings WHERE key = 'initial_bankroll'")
    init_row = cursor.fetchone()
    initial_bankroll = float(init_row['value']) if init_row else 100000.0
    
    cursor.execute("SELECT value FROM settings WHERE key = 'currency'")
    curr_row = cursor.fetchone()
    currency = curr_row['value'] if curr_row else "Ft"

    cursor.execute("SELECT * FROM bets ORDER BY placed_at ASC")
    bets = cursor.fetchall()
    
    total_bets = len(bets)
    pending_bets = 0
    won_bets = 0
    lost_bets = 0
    void_bets = 0
    total_staked = 0.0
    total_pnl = 0.0
    pending_staked = 0.0
    
    chart_points = [{"label": "Start", "pnl": 0.0, "bankroll": initial_bankroll}]
    cumulative_pnl = 0.0
    
    for b in bets:
        status = b['status']
        stake = float(b['stake_amount'] or 0.0)
        pnl = float(b['pnl'] or 0.0)
        
        if status == 'PENDING':
            pending_bets += 1
            pending_staked += stake
        elif status == 'WON':
            won_bets += 1
            total_staked += stake
            total_pnl += pnl
            cumulative_pnl += pnl
            chart_points.append({
                "label": b['match_date'][:10] if b['match_date'] else "Match",
                "pnl": round(cumulative_pnl, 2),
                "bankroll": round(initial_bankroll + cumulative_pnl, 2)
            })
        elif status == 'LOST':
            lost_bets += 1
            total_staked += stake
            total_pnl += pnl
            cumulative_pnl += pnl
            chart_points.append({
                "label": b['match_date'][:10] if b['match_date'] else "Match",
                "pnl": round(cumulative_pnl, 2),
                "bankroll": round(initial_bankroll + cumulative_pnl, 2)
            })
        elif status == 'VOID':
            void_bets += 1
            chart_points.append({
                "label": b['match_date'][:10] if b['match_date'] else "Match",
                "pnl": round(cumulative_pnl, 2),
                "bankroll": round(initial_bankroll + cumulative_pnl, 2)
            })
            
    settled_bets = won_bets + lost_bets
    win_rate = (won_bets / settled_bets * 100) if settled_bets > 0 else 0.0
    roi = (total_pnl / total_staked * 100) if total_staked > 0 else 0.0
    current_bankroll = initial_bankroll + total_pnl
    conn.close()
    
    return jsonify({
        "success": True,
        "initial_bankroll": initial_bankroll,
        "current_bankroll": current_bankroll,
        "currency": currency,
        "total_bets": total_bets,
        "pending_bets": pending_bets,
        "pending_staked": pending_staked,
        "settled_bets": settled_bets,
        "won_bets": won_bets,
        "lost_bets": lost_bets,
        "void_bets": void_bets,
        "win_rate": round(win_rate, 2),
        "total_staked": round(total_staked, 2),
        "total_pnl": round(total_pnl, 2),
        "roi": round(roi, 2),
        "chart_points": chart_points
    })

@app.route('/api/bankroll/bets', methods=['GET'])
def get_bankroll_bets():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM bets ORDER BY id DESC")
    rows = cursor.fetchall()
    bets = [dict(r) for r in rows]
    conn.close()
    return jsonify({"success": True, "bets": bets})

@app.route('/api/bankroll/add', methods=['POST'])
def add_bankroll_bet():
    data = request.json or {}
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO bets (
            match_date, match_id, match_name, selection, 
            odds_placed, odds_closing, stake_amount, stake_pct, ev_pct, status
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'PENDING')
    """, (
        data.get('match_date', '-'),
        data.get('match_id', 0),
        data.get('match_name', 'Unknown'),
        data.get('selection', '-'),
        float(data.get('odds_placed', 1.0)),
        float(data.get('odds_closing', 0.0) or 0.0),
        float(data.get('stake_amount', 0.0)),
        float(data.get('stake_pct', 0.0)),
        float(data.get('ev_pct', 0.0))
    ))
    conn.commit()
    conn.close()
    return jsonify({"success": True, "message": "Bet successfully recorded in portfolio!"})

@app.route('/api/bankroll/update-closing-odds', methods=['POST'])
def update_closing_odds():
    data = request.json or {}
    bet_id = data.get('bet_id')
    odds_closing = float(data.get('odds_closing') or 0.0)
    
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("UPDATE bets SET odds_closing = ? WHERE id = ?", (odds_closing, bet_id))
    conn.commit()
    conn.close()
    return jsonify({"success": True, "message": "Closing odds successfully saved."})

@app.route('/api/bankroll/settle', methods=['POST'])
def settle_bankroll_bet():
    data = request.json or {}
    bet_id = data.get('bet_id')
    result = data.get('result', '').upper()
    odds_closing = float(data.get('odds_closing') or 0.0)
    
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM bets WHERE id = ?", (bet_id,))
    bet = cursor.fetchone()
    if not bet:
        conn.close()
        return jsonify({"success": False, "message": "Bet not found!"})
    
    stake = float(bet['stake_amount'])
    odds = float(bet['odds_placed'])
    
    if result == 'WON': pnl = stake * (odds - 1.0)
    elif result == 'LOST': pnl = -stake
    elif result == 'VOID': pnl = 0.0
    else:
        conn.close()
        return jsonify({"success": False, "message": "Invalid result!"})
        
    settled_at = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cursor.execute("""
        UPDATE bets 
        SET status = ?, pnl = ?, odds_closing = CASE WHEN ? > 0 THEN ? ELSE odds_closing END, settled_at = ?
        WHERE id = ?
    """, (result, round(pnl, 2), odds_closing, odds_closing, settled_at, bet_id))
    conn.commit()
    conn.close()
    return jsonify({"success": True, "message": f"Bet settled: {result}"})

@app.route('/api/bankroll/delete', methods=['POST'])
def delete_bankroll_bet():
    data = request.json or {}
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM bets WHERE id = ?", (data.get('bet_id'),))
    conn.commit()
    conn.close()
    return jsonify({"success": True, "message": "Bet deleted."})

@app.route('/api/bankroll/settings', methods=['POST'])
def update_bankroll_settings():
    data = request.json or {}
    init_val = data.get('initial_bankroll')
    conn = get_db_connection()
    cursor = conn.cursor()
    if init_val is not None:
        cursor.execute("UPDATE settings SET value = ? WHERE key = 'initial_bankroll'", (str(float(init_val)),))
    conn.commit()
    conn.close()
    return jsonify({"success": True, "message": "Initial bankroll set."})

def auto_settle_pending_bets_internal():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM bets WHERE status = 'PENDING'")
    pending_bets = cursor.fetchall()
    
    if not pending_bets:
        conn.close()
        return "No pending bets to settle."
        
    headers = {'User-Agent': 'Mozilla/5.0'}
    url_league = "https://www.fotmob.com/api/data/leagues?id=47&ccode3=HUN"
    
    try:
        res = requests.get(url_league, headers=headers, timeout=10).json()
    except Exception as e:
        conn.close()
        return f"Failed to reach FotMob API for results: {e}"

    def extract_matches(obj):
        found = []
        if isinstance(obj, dict):
            if ('home' in obj or 'homeTeam' in obj) and ('away' in obj or 'awayTeam' in obj) and 'id' in obj:
                found.append(obj)
            for v in obj.values(): found.extend(extract_matches(v))
        elif isinstance(obj, list):
            for item in obj: found.extend(extract_matches(item))
        return found

    all_matches = {str(m.get('id')): m for m in extract_matches(res)}
    settled_count = 0
    now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    for b in pending_bets:
        mid_str = str(b['match_id'])
        if mid_str in all_matches:
            m = all_matches[mid_str]
            status_obj = m.get('status', {})
            
            if status_obj.get('finished', False):
                h_score = None
                a_score = None
                score_str = status_obj.get('scoreStr')
                if score_str and '-' in score_str:
                    try:
                        parts = score_str.split('-')
                        h_score, a_score = int(parts[0].strip()), int(parts[1].strip())
                    except Exception: pass
                
                if h_score is None:
                    h_obj = m.get('home') if isinstance(m.get('home'), dict) else m.get('homeTeam', {})
                    a_obj = m.get('away') if isinstance(m.get('away'), dict) else m.get('awayTeam', {})
                    h_score = h_obj.get('score')
                    a_score = a_obj.get('score')

                if h_score is not None and a_score is not None:
                    h_score, a_score = int(h_score), int(a_score)
                    actual_result = 'X'
                    if h_score > a_score: actual_result = '1'
                    elif a_score > h_score: actual_result = '2'

                    sel = str(b['selection'])
                    is_win = False
                    if '1' in sel and actual_result == '1': is_win = True
                    elif ('x' in sel.lower() or 'döntetlen' in sel.lower()) and actual_result == 'X': is_win = True
                    elif '2' in sel and actual_result == '2': is_win = True

                    stake = float(b['stake_amount'])
                    odds = float(b['odds_placed'])
                    res_status = 'WON' if is_win else 'LOST'
                    pnl = stake * (odds - 1.0) if is_win else -stake

                    cursor.execute("""
                        UPDATE bets SET status = ?, pnl = ?, settled_at = ? WHERE id = ?
                    """, (res_status, round(pnl, 2), now_str, b['id']))
                    settled_count += 1

    conn.commit()
    conn.close()
    return f"{settled_count} matches automatically settled based on results."

@app.route('/api/bankroll/auto-settle', methods=['POST'])
def auto_settle_route():
    log = auto_settle_pending_bets_internal()
    return jsonify({"success": True, "log": log})

@app.route('/api/master-sync', methods=['POST'])
def master_sync():
    logs = []
    logs.append("==================================================")
    logs.append("🚀 INITIATING WEEKLY MASTER SYNC")
    logs.append("==================================================\n")

    logs.append("[1/3] Automatically settling pending bets...")
    settle_log = auto_settle_pending_bets_internal()
    logs.append(f" -> {settle_log}\n")

    logs.append("[2/3] Executing database pipeline (!data_collecting -> !data_processing -> !Elo_rest)...")
    pipeline_steps = ["!data_collecting.py", "!data_processing.py", "!Elo_rest.py"]
    for script in pipeline_steps:
        logs.append(f"--- Running: {script} ---")
        ok, out = run_script(script)
        logs.append(out.strip())
        if not ok:
            logs.append(f"\n[!] Pipeline aborted at step: {script}.")
            return jsonify({"success": False, "log": "\n".join(logs)})
    logs.append(" -> Database and Elo matrix successfully updated.\n")

    logs.append("[3/3] Analyzing upcoming matches and evaluating odds (!live_predictor.py)...")
    ok, out = run_script("!live_predictor.py")
    logs.append(out.strip())
    if not ok:
        logs.append("\n[!] Error during prediction execution.")
        return jsonify({"success": False, "log": "\n".join(logs)})

    logs.append("\n==================================================")
    logs.append("✅ MASTER SYNC SUCCESSFULLY COMPLETED!")
    logs.append("Matches settled, database updated, new predictions ready!")
    logs.append("==================================================")

    return jsonify({"success": True, "log": "\n".join(logs)})

@app.route('/api/retrain-model', methods=['POST'])
def retrain_model():
    success, output = run_script("!ml_predict_probabilities.py")
    return jsonify({"success": success, "log": output})

if __name__ == '__main__':
    print("Dashboard started at: http://127.0.0.1:5000")
    app.run(debug=True, port=5000)