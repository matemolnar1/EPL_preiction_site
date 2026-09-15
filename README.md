# Premier League Quant Betting Pipeline & Dashboard

**Author:** Molnár Máté

An end-to-end data engineering, machine learning, and web dashboard pipeline designed to identify mathematical edge (Expected Value) in English Premier League betting markets. This portfolio project was developed utilizing AI-assisted rapid prototyping to focus heavily on high-level system architecture, automated data pipelines, and quantitative finance principles.

## ⚠️ Disclaimer
**This project is strictly for educational and portfolio demonstration purposes.** The current model and Expected Value (EV) calculations are experimental and are **not reliable enough for real-money betting**. The system is designed to simulate a quantitative workflow, not to serve as financial advice. Do not risk actual capital based on these outputs.

## About the Project
This system automates the full lifecycle of a quantitative sports betting model. It extracts historical and live match features, unpivots them into continuous chronological timelines, calculates advanced dynamic metrics (like Elo ratings and rest-day differentials), and feeds them into an XGBoost classifier. A Flask-based frontend is used to visualize predictions, log bets, and simulate a fractional Kelly Criterion bankroll.

## System Architecture

*   **Data Ingestion (`!data_collecting.py`):** Automatically scrapes and aggregates detailed match features (xG, xGOT, touches in opponent box, momentum) from the FotMob API.
*   **Feature Engineering (`!data_processing.py` & `!Elo_rest.py`):** Unpivots raw data to create continuous team timelines. Calculates pre-match Exponential Moving Averages (EMA) and dynamic Elo ratings accounting for home-field advantage while strictly preventing data leakage.
*   **Machine Learning (`!ml_predict_probabilities.py`):** Trains an XGBoost classifier on historical data, outputting soft probabilities for Home/Draw/Away outcomes. 
*   **Live Operations (`!app.py` & `!live_predictor.py`):** A Flask backend that orchestrates a "Master Sync" process. It fetches real-time bookmaker odds, compares them against the model's probabilities to find Value Bets, and calculates optimal fractional Kelly stakes.
*   **Backtesting (`!ev_backtester.py`):** Simulates historical profitability by merging past predictions with historical closing odds (sourced from football-data.co.uk), demonstrating an understanding of Expected Value and risk management. Note: Historical odds CSVs are only required for backtesting, not for live weekly operations.
## Model Performance & Accuracy
The current XGBoost classification model operates at a **46.58% accuracy** on the hold-out validation set. 

While this number might seem low in a standard machine learning context, it is important to note that predicting a 3-way football market (Home/Draw/Away) is notoriously difficult; random guessing yields only 33.3%. In quantitative sports betting, the objective is rarely to achieve astronomical raw accuracy. Instead, the goal is to build a model that accurately estimates true probabilities, allowing the system to identify inefficiencies and Expected Value (EV) when compared against a bookmaker's implied odds.

## Setup & Installation

1.  **Clone the Repository:** Download the project files into an isolated directory.
2.  **Environment Setup:** Ensure Python is installed. Create a virtual environment and install the required packages:
    ```bash
    pip install -r requirements.txt
    ```
    *(Note: Key dependencies include `pandas`, `numpy`, `xgboost`, `scikit-learn`, `requests`, and `Flask`.)*
3.  **Launch the Dashboard:** Run the web application locally:
    ```bash
    python !app.py
    ```
4.  **Initialize the System:** Open `http://127.0.0.1:5000` in your web browser. Click the **Master Sync** button on the dashboard to trigger the automated data collection, pipeline execution, and live prediction engine.

   ### Note on Initial Setup (Cold Start)
Because this application is entirely self-hosted and generates its own local SQLite databases and JSON cache files, the very first time you launch the dashboard, it is in a "cold" state. You may need to trigger the **Master Sync** or data collection pipelines once or twice from the UI to fully generate the necessary backend files. Once the initial databases are populated, subsequent syncs will run seamlessly in a single click.
