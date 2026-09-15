import os
import joblib
import pandas as pd
import numpy as np

def analyze_feature_importance(model_path="xgboost_epl_model.pkl"):
    """Extracts and calculates feature importance metrics from the trained XGBoost model."""
    if not os.path.exists(model_path):
        model_path = "ml_foci_agy.pkl"
        if not os.path.exists(model_path):
            print(f"Error: Trained model not found at {model_path}!")
            return None
    
    model = joblib.load(model_path)
    booster = model.get_booster()

    gain_scores = booster.get_score(importance_type='gain')
    weight_scores = booster.get_score(importance_type='weight')
    cover_scores = booster.get_score(importance_type='cover')

    feature_names = getattr(model, 'feature_names_in_', list(gain_scores.keys()))

    importance_data = []
    for feat in feature_names:
        importance_data.append({
            'Feature': feat,
            'Gain': gain_scores.get(feat, 0.0),
            'Weight': weight_scores.get(feat, 0),
            'Cover': cover_scores.get(feat, 0.0)
        })

    df_imp = pd.DataFrame(importance_data)

    total_gain = df_imp['Gain'].sum()
    if total_gain > 0:
        df_imp['Fontossag_%'] = (df_imp['Gain'] / total_gain) * 100
    else:
        df_imp['Fontossag_%'] = 0.0

    df_imp.sort_values(by='Gain', ascending=False, inplace=True)
    df_imp.reset_index(drop=True, inplace=True)
    return df_imp


if __name__ == "__main__":
    print("Loading model and analyzing decision trees...")
    df_imp = analyze_feature_importance()
    
    if df_imp is not None:
        print("\n" + "=" * 85)
        print(f"{'#':<3} | {'FEATURE NAME':<35} | {'IMPORTANCE %':>12} | {'GAIN VALUE':>12} | {'USAGE (SPLIT)':>15}")
        print("=" * 85)

        for idx, r in df_imp.head(25).iterrows():
            print(f"{idx+1:<3} | {r['Feature']:<35} | {r['Fontossag_%']:>11.2f}% | {r['Gain']:>12.2f} | {int(r['Weight']):>15}")

        print("=" * 85)

        df_imp.to_csv("feature_importances.csv", index=False, encoding='utf-8-sig')
        print(f"\n[OK] Full list ({len(df_imp)} features) saved to: feature_importances.csv")

        try:
            import matplotlib.pyplot as plt
            top_plot = df_imp.head(15).sort_values(by='Gain', ascending=True)
            
            plt.figure(figsize=(10, 6))
            plt.barh(top_plot['Feature'], top_plot['Fontossag_%'], color='#10b981')
            plt.xlabel('Relative Importance (% based on Gain)')
            plt.title('Top 15 Most Important Decision Features in XGBoost Model')
            plt.tight_layout()
            plt.savefig('feature_importance_top15.png', dpi=200)
            print("[OK] Graph saved as image: feature_importance_top15.png")
        except ImportError:
            print("Matplotlib not installed; skipping chart generation.")