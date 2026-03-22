"""
Test model.joblib on data/training.parquet.

Evaluates all 5 sub-models (catastrophe classifier, outgassing impact
classifier, mineral value regressor, yield regressor, delay regressor)
and then runs the full bidding strategy on sample rounds.
"""

import sys
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    classification_report,
    log_loss,
    mean_absolute_error,
    r2_score,
    roc_auc_score,
)

sys.path.insert(0, "strategies")
from strategy import _engineer_features, _model, price_asteroids

CATASTROPHE_MAP = {
    "none": 0,
    "void_rock": 1,
    "structural_collapse": 2,
    "toxic_outgassing": 3,
}
TARGET_COLS = [
    "mineral_value",
    "extraction_yield",
    "extraction_delay",
    "catastrophe_type",
    "toxic_outgassing_impact",
]


def evaluate_models(df_raw):
    model = _model
    feat_cols = model["feature_columns"]
    df_feat = df_raw.copy()

    y_cat = df_raw["catastrophe_type"].map(CATASTROPHE_MAP).values
    y_impact = df_raw["toxic_outgassing_impact"].values
    clean = (df_raw["catastrophe_type"] == "none") & (y_impact == 0)
    y_val = df_raw.loc[clean, "mineral_value"].values
    y_yld = df_raw.loc[clean, "extraction_yield"].values
    y_del = df_raw["extraction_delay"].values

    X_all = _engineer_features(df_feat, feat_cols)

    print(f"Dataset: {len(df_raw)} rows, {X_all.shape[1]} features")
    print(f"Clean rows (no catastrophe, no outgassing): {clean.sum()}")
    print()

    # 1. Catastrophe classifier
    print("=" * 60)
    print("1. Catastrophe Classifier (multiclass)")
    print("=" * 60)
    cat_proba = model["catastrophe_model"].predict(X_all)
    cat_pred = cat_proba.argmax(axis=1)
    print(f"  Log Loss : {log_loss(y_cat, cat_proba):.4f}")
    print(f"  Accuracy : {(cat_pred == y_cat).mean():.4f}")
    print(
        classification_report(
            y_cat,
            cat_pred,
            target_names=["none", "void_rock", "collapse", "outgas"],
            zero_division=0,
        )
    )

    # 2. Outgassing impact classifier
    print("=" * 60)
    print("2. Outgassing Impact Classifier (binary)")
    print("=" * 60)
    impact_proba = model["outgassing_model"].predict(X_all)
    print(f"  ROC-AUC  : {roc_auc_score(y_impact, impact_proba):.4f}")
    print(f"  Log Loss : {log_loss(y_impact, impact_proba):.4f}")
    print()

    # 3. Mineral value regressor (clean rows only)
    print("=" * 60)
    print("3. Mineral Value Regressor (clean rows)")
    print("=" * 60)
    X_cln = X_all.loc[clean].reset_index(drop=True)
    val_pred = np.maximum(model["value_model"].predict(X_cln), 0.0)
    print(f"  MAE : {mean_absolute_error(y_val, val_pred):.2f}")
    print(f"  R²  : {r2_score(y_val, val_pred):.4f}")
    print()

    # 4. Extraction yield regressor (clean rows only)
    print("=" * 60)
    print("4. Extraction Yield Regressor (clean rows)")
    print("=" * 60)
    yld_pred = np.clip(model["yield_model"].predict(X_cln), 0.0, 1.5)
    print(f"  MAE : {mean_absolute_error(y_yld, yld_pred):.4f}")
    print(f"  R²  : {r2_score(y_yld, yld_pred):.4f}")
    print()

    # 5. Extraction delay regressor (all rows)
    print("=" * 60)
    print("5. Extraction Delay Regressor (all rows)")
    print("=" * 60)
    del_pred = np.clip(model["delay_model"].predict(X_all), 1.0, 20.0)
    print(f"  MAE : {mean_absolute_error(y_del, del_pred):.2f}")
    print(f"  R²  : {r2_score(y_del, del_pred):.4f}")
    print()


def test_bidding(df_raw, n_rounds=5, batch_size=10):
    print("=" * 60)
    print(f"Bidding Strategy Test ({n_rounds} simulated rounds)")
    print("=" * 60)

    rng = np.random.default_rng(42)
    capital = 10_000.0

    for rnd in range(1, n_rounds + 1):
        sample = df_raw.sample(n=batch_size, random_state=rng.integers(1e9))
        batch = []
        for _, row in sample.iterrows():
            features = row.drop(TARGET_COLS).to_dict()
            features.pop("asteroid_id", None)
            batch.append(features)

        bids = price_asteroids(
            batch,
            capital=capital,
            round_info={
                "round_number": rnd,
                "total_rounds": 100,
                "sector_name": "Outer Rim",
                "asteroids_this_round": batch_size,
                "risk_free_rate": 0.002,
                "num_active_competitors": 5,
                "pending_revenue": 0.0,
                "num_pending_extractions": 0,
                "previous_round": None,
                "market_history": None,
            },
        )

        print(f"\n  Round {rnd} | Capital: ${capital:,.0f}")
        print(f"  {'Idx':>3}  {'Bid':>8}  {'MineralVal':>10}  {'Yield':>6}  {'Catastrophe':>18}")
        print(f"  {'---':>3}  {'--------':>8}  {'----------':>10}  {'------':>6}  {'------------------':>18}")
        for i, bid in enumerate(bids):
            row = sample.iloc[i]
            print(
                f"  {i:3d}  {bid:8.2f}  {row['mineral_value']:10.2f}  "
                f"{row['extraction_yield']:6.3f}  {row['catastrophe_type']:>18}"
            )
        total_bid = sum(bids)
        n_bids = sum(1 for b in bids if b > 0)
        print(f"  Total bid: ${total_bid:,.2f} ({n_bids}/{batch_size} asteroids)")


def main():
    print(f"Model loaded at import: {list(_model.keys())}")
    print()

    print("Loading data/training.parquet...")
    df = pd.read_parquet("data/training.parquet")
    print()

    evaluate_models(df)
    test_bidding(df)


if __name__ == "__main__":
    main()