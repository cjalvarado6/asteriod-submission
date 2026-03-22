"""
Training pipeline v3 — best of v1 and v2.

Combines v2's superior feature set (including ai_valuation_estimate and
engineered features) with v1's 5-model architecture (adds binary outgassing
classifier), native categorical handling, and proper early-stopping CV.

Models:
  1. Catastrophe multiclass classifier  (none / void_rock / collapse / outgassing)
  2. Outgassing impact binary classifier (from v1 — missing in v2)
  3. Mineral value regressor              (trained on clean rows only)
  4. Extraction yield regressor            (trained on clean rows only)
  5. Extraction delay regressor            (trained on all rows)
"""
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import (
    classification_report,
    log_loss,
    mean_absolute_error,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import KFold, StratifiedKFold

DATA_PATH = Path("data/training.parquet")
MODEL_PATH = Path("model3.joblib")

TARGET_COLS = [
    "mineral_value",
    "extraction_yield",
    "extraction_delay",
    "catastrophe_type",
    "toxic_outgassing_impact",
]

DROP_COLS = [
    "asteroid_id",
    "time_period",
    "lucky_number",
    "media_hype_score",
    "social_sentiment_score",
    "cluster_id",
    "ai_valuation_estimate",
    "analyst_consensus_estimate",
]

CATEGORICAL_COLS = ["spectral_class", "belt_region", "probe_type"]

SPECTRAL_MAP = {"C-type": 0, "S-type": 1, "M-type": 2, "X-type": 3}
BELT_MAP = {"inner": 0, "main": 1, "outer": 2}
PROBE_MAP = {"passive": 0, "active_flyby": 1, "landing": 2, "drill_core": 3}
CATASTROPHE_MAP = {
    "none": 0,
    "void_rock": 1,
    "structural_collapse": 2,
    "toxic_outgassing": 3,
}

MINERAL_PAIRS = [
    ("mineral_signature_iron", "mineral_price_iron", "iron"),
    ("mineral_signature_nickel", "mineral_price_nickel", "nickel"),
    ("mineral_signature_cobalt", "mineral_price_cobalt", "cobalt"),
    ("mineral_signature_platinum", "mineral_price_platinum", "platinum"),
    ("mineral_signature_rare_earth", "mineral_price_rare_earth", "rare_earth"),
]


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Transform raw asteroid data into model features.

    This logic must be kept in sync with strategy3.py.
    """
    df = df.copy()

    # Categoricals → integer codes (LightGBM will be told they're categorical)
    df["spectral_class"] = df["spectral_class"].map(SPECTRAL_MAP).fillna(1).astype(int)
    df["belt_region"] = df["belt_region"].map(BELT_MAP).fillna(1).astype(int)
    df["probe_type"] = df["probe_type"].map(PROBE_MAP).fillna(0).astype(int)

    # Mineral signature × price interactions (from v2)
    for sig_col, price_col, mineral in MINERAL_PAIRS:
        df[f"mineral_val_{mineral}"] = df[sig_col] * df[price_col]
    df["mineral_val_water"] = df["water_ice_fraction"] * df["mineral_price_water"]

    mineral_val_cols = [
        "mineral_val_iron",
        "mineral_val_nickel",
        "mineral_val_cobalt",
        "mineral_val_platinum",
        "mineral_val_rare_earth",
        "mineral_val_water",
    ]
    df["total_mineral_estimate"] = df[mineral_val_cols].sum(axis=1)

    # Catastrophe risk composite (from v2)
    df["risk_score"] = (
        (1 - df["structural_integrity"])
        * df["porosity"]
        * (1 + df["volatile_content"])
    )

    # Operational feasibility (from v2)
    df["access_difficulty_ratio"] = df["accessibility_score"] / (
        df["extraction_difficulty"] + 0.01
    )

    # Survey quality composite (from v2)
    df["survey_quality"] = (
        df["survey_confidence"]
        * df["data_completeness"]
        * (1 - df["conflicting_results"] * 0.3)
    )

    cols_to_drop = [c for c in DROP_COLS + TARGET_COLS if c in df.columns]
    df.drop(columns=cols_to_drop, inplace=True)

    return df


def train_lgb(
    X: pd.DataFrame,
    y: np.ndarray,
    params: dict,
    cat_features: list[str] | None = None,
    n_splits: int = 5,
    stratify: np.ndarray | None = None,
    patience: int = 50,
    name: str = "Model",
):
    """Train LightGBM with CV and early stopping, return final model + OOF."""
    print(f"\n{'=' * 60}")
    print(f"Training: {name}")
    print(f"{'=' * 60}")
    print(f"  Samples: {len(X)}, Features: {X.shape[1]}")

    if stratify is not None:
        folder = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        splits = folder.split(X, stratify)
    else:
        folder = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        splits = folder.split(X)

    best_iters = []
    oof = None

    for fold, (tr_idx, va_idx) in enumerate(splits):
        ds_tr = lgb.Dataset(
            X.iloc[tr_idx], label=y[tr_idx], categorical_feature=cat_features
        )
        ds_va = lgb.Dataset(X.iloc[va_idx], label=y[va_idx], reference=ds_tr)

        m = lgb.train(
            params,
            ds_tr,
            num_boost_round=2000,
            valid_sets=[ds_va],
            callbacks=[lgb.early_stopping(patience), lgb.log_evaluation(0)],
        )

        preds = m.predict(X.iloc[va_idx])
        best_iters.append(m.best_iteration)

        if oof is None:
            shape = (len(X), preds.shape[1]) if preds.ndim == 2 else (len(X),)
            oof = np.zeros(shape)
        oof[va_idx] = preds

        print(f"  Fold {fold + 1}: best_iteration = {m.best_iteration}")

    n_rounds = int(np.median(best_iters))
    print(f"  Median best iteration: {n_rounds}")

    ds_full = lgb.Dataset(X, label=y, categorical_feature=cat_features)
    final = lgb.train(params, ds_full, num_boost_round=n_rounds)
    return final, oof


def main():
    print("Loading data...")
    raw = pd.read_parquet(DATA_PATH)
    print(f"  {len(raw)} rows, {len(raw.columns)} columns\n")

    # ---- targets ----
    y_cat = raw["catastrophe_type"].map(CATASTROPHE_MAP).values
    y_impact = raw["toxic_outgassing_impact"].values
    clean = (raw["catastrophe_type"] == "none") & (raw["toxic_outgassing_impact"] == 0)
    y_val = raw.loc[clean, "mineral_value"].values
    y_yld = raw.loc[clean, "extraction_yield"].values
    y_del = raw["extraction_delay"].values

    print(f"Clean rows (no catastrophe, no outgassing impact): {clean.sum()}")
    print(f"Outgassing impacts: {y_impact.sum()}")

    # ---- features ----
    X_all = engineer_features(raw)
    X_cln = X_all.loc[clean].reset_index(drop=True)
    X_all = X_all.reset_index(drop=True)

    feat_cols = X_all.columns.tolist()
    print(f"Features: {len(feat_cols)} columns\n")

    # ==== 1. Catastrophe classifier (multiclass) ====
    m_cat, oof_cat = train_lgb(
        X_all,
        y_cat,
        {
            "objective": "multiclass",
            "num_class": 4,
            "metric": "multi_logloss",
            "is_unbalance": True,
            "learning_rate": 0.05,
            "num_leaves": 31,
            "min_child_samples": 20,
            "feature_fraction": 0.8,
            "bagging_fraction": 0.8,
            "bagging_freq": 5,
            "reg_alpha": 0.1,
            "reg_lambda": 1.0,
            "verbose": -1,
            "seed": 42,
        },
        cat_features=CATEGORICAL_COLS,
        stratify=y_cat,
        patience=100,
        name="Catastrophe Classifier (multiclass)",
    )
    print(f"  Log Loss : {log_loss(y_cat, oof_cat):.4f}")
    print(f"  Accuracy : {(oof_cat.argmax(1) == y_cat).mean():.4f}")
    print(
        classification_report(
            y_cat,
            oof_cat.argmax(1),
            target_names=["none", "void_rock", "collapse", "outgas"],
            zero_division=0,
        )
    )
    for label, lname in enumerate(["none", "void_rock", "collapse", "outgas"]):
        mask = y_cat == label
        print(
            f"  P({lname:10s}) — actual: {oof_cat[mask, label].mean():.3f},"
            f"  others: {oof_cat[~mask, label].mean():.3f}"
        )

    # ==== 2. Outgassing impact classifier (binary) — from v1 ====
    m_impact, oof_impact = train_lgb(
        X_all,
        y_impact,
        {
            "objective": "binary",
            "metric": "binary_logloss",
            "is_unbalance": True,
            "learning_rate": 0.05,
            "num_leaves": 31,
            "min_child_samples": 20,
            "feature_fraction": 0.8,
            "bagging_fraction": 0.8,
            "bagging_freq": 5,
            "reg_alpha": 0.1,
            "reg_lambda": 1.0,
            "verbose": -1,
            "seed": 42,
        },
        cat_features=CATEGORICAL_COLS,
        stratify=y_impact,
        patience=100,
        name="Outgassing Impact Classifier (binary)",
    )
    print(f"  ROC-AUC  : {roc_auc_score(y_impact, oof_impact):.4f}")
    print(f"  Log Loss : {log_loss(y_impact, oof_impact):.4f}")

    # ==== 3. Mineral value regressor ====
    m_val, oof_val = train_lgb(
        X_cln,
        y_val,
        {
            "objective": "regression",
            "metric": "mae",
            "learning_rate": 0.05,
            "num_leaves": 127,
            "min_child_samples": 20,
            "feature_fraction": 0.8,
            "bagging_fraction": 0.8,
            "bagging_freq": 5,
            "reg_alpha": 0.1,
            "reg_lambda": 0.1,
            "verbose": -1,
            "seed": 42,
        },
        cat_features=CATEGORICAL_COLS,
        name="Mineral Value Regressor",
    )
    print(f"  MAE : {mean_absolute_error(y_val, oof_val):.2f}")
    print(f"  R²  : {r2_score(y_val, oof_val):.4f}")

    # ==== 4. Extraction yield regressor ====
    m_yld, oof_yld = train_lgb(
        X_cln,
        y_yld,
        {
            "objective": "regression",
            "metric": "mae",
            "learning_rate": 0.05,
            "num_leaves": 31,
            "min_child_samples": 20,
            "feature_fraction": 0.8,
            "bagging_fraction": 0.8,
            "bagging_freq": 5,
            "reg_alpha": 0.1,
            "reg_lambda": 1.0,
            "verbose": -1,
            "seed": 42,
        },
        cat_features=CATEGORICAL_COLS,
        name="Extraction Yield Regressor",
    )
    print(f"  MAE : {mean_absolute_error(y_yld, oof_yld):.4f}")
    print(f"  R²  : {r2_score(y_yld, oof_yld):.4f}")

    # ==== 5. Extraction delay regressor ====
    m_del, oof_del = train_lgb(
        X_all,
        y_del,
        {
            "objective": "regression",
            "metric": "mae",
            "learning_rate": 0.05,
            "num_leaves": 31,
            "min_child_samples": 30,
            "feature_fraction": 0.8,
            "bagging_fraction": 0.8,
            "bagging_freq": 5,
            "reg_alpha": 0.1,
            "reg_lambda": 0.1,
            "verbose": -1,
            "seed": 42,
        },
        cat_features=CATEGORICAL_COLS,
        name="Extraction Delay Regressor",
    )
    print(f"  MAE : {mean_absolute_error(y_del, oof_del):.2f}")
    print(f"  R²  : {r2_score(y_del, oof_del):.4f}")

    # ---- feature importance (value model) ----
    print(f"\n{'=' * 60}")
    print("Top 20 features by gain (Mineral Value model)")
    print(f"{'=' * 60}")
    imp = m_val.feature_importance(importance_type="gain")
    for feat_name, gain in sorted(zip(feat_cols, imp), key=lambda x: -x[1])[:20]:
        print(f"  {feat_name:40s} {gain:>12.1f}")

    # ---- save ----
    bundle = {
        "catastrophe_model": m_cat,
        "outgassing_model": m_impact,
        "value_model": m_val,
        "yield_model": m_yld,
        "delay_model": m_del,
        "feature_columns": feat_cols,
        "categorical_columns": CATEGORICAL_COLS,
        "category_maps": {
            "spectral_class": SPECTRAL_MAP,
            "belt_region": BELT_MAP,
            "probe_type": PROBE_MAP,
        },
    }
    joblib.dump(bundle, MODEL_PATH, compress=3)
    sz = MODEL_PATH.stat().st_size / 1024 / 1024
    print(f"\nSaved {MODEL_PATH} ({sz:.1f} MB)")
    if sz > 50:
        print("WARNING: model file exceeds the 50 MB submission limit!")


if __name__ == "__main__":
    main()
