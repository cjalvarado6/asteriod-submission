"""
Asteroid Auction Strategy v3 — best of v1 and v2.

Combines v2's superior ML predictions (ai_valuation_estimate features,
engineered features) with v1's battle-tested strategy logic (outgassing
model, cluster concentration, economic cycle adjustment, tiered capital
management).

Models loaded from model.joblib:
  1. Catastrophe multiclass classifier → P(none, void, collapse, outgas)
  2. Outgassing impact binary classifier → P(impacted by neighbor outgassing)
  3. Mineral value regressor → E[mineral_value | safe]
  4. Extraction yield regressor → E[extraction_yield | safe]
  5. Extraction delay regressor → E[extraction_delay]
"""

import os
import joblib
import lightgbm  # noqa: F401 — needed for joblib to deserialise Booster objects

STRATEGY_NAME = "Deep Rock Mining v3"

_model_path = os.path.join(os.path.dirname(__file__), "model.joblib")
_model = joblib.load(_model_path)

# --- Tunable constants (from v1, proven in competition structure) ------------

WINNERS_CURSE_BASE = 0.50
WINNERS_CURSE_PER_COMPETITOR = 0.03
MAX_BID_FRACTION = 0.12
MAX_PORTFOLIO_FRACTION = 0.70
MIN_BID = 15.0
LOW_CAPITAL_THRESHOLD = 100.0
CRITICAL_CAPITAL_THRESHOLD = 50.0
CLUSTER_PENALTY_FACTOR = 0.25

_SPECTRAL_MAP = {"C-type": 0, "S-type": 1, "M-type": 2, "X-type": 3}
_BELT_MAP = {"inner": 0, "main": 1, "outer": 2}
_PROBE_MAP = {"passive": 0, "active_flyby": 1, "landing": 2, "drill_core": 3}

_MINERAL_PAIRS = [
    ("mineral_signature_iron", "mineral_price_iron", "iron"),
    ("mineral_signature_nickel", "mineral_price_nickel", "nickel"),
    ("mineral_signature_cobalt", "mineral_price_cobalt", "cobalt"),
    ("mineral_signature_platinum", "mineral_price_platinum", "platinum"),
    ("mineral_signature_rare_earth", "mineral_price_rare_earth", "rare_earth"),
]

_DROP_COLS = {
    "asteroid_id",
    "time_period",
    "lucky_number",
    "media_hype_score",
    "social_sentiment_score",
    "cluster_id",
    "ai_valuation_estimate",
    "analyst_consensus_estimate",
}


def _engineer_features(df, feature_columns):
    """Apply identical feature transforms as used in training.

    Returns a DataFrame with columns ordered to match the model's expectations.
    """
    df["spectral_class"] = df["spectral_class"].map(_SPECTRAL_MAP).fillna(1).astype(int)
    df["belt_region"] = df["belt_region"].map(_BELT_MAP).fillna(1).astype(int)
    df["probe_type"] = df["probe_type"].map(_PROBE_MAP).fillna(0).astype(int)

    for sig_col, price_col, mineral in _MINERAL_PAIRS:
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

    df["risk_score"] = (
        (1 - df["structural_integrity"])
        * df["porosity"]
        * (1 + df["volatile_content"])
    )

    df["access_difficulty_ratio"] = df["accessibility_score"] / (
        df["extraction_difficulty"] + 0.01
    )

    df["survey_quality"] = (
        df["survey_confidence"]
        * df["data_completeness"]
        * (1 - df["conflicting_results"] * 0.3)
    )

    for col in _DROP_COLS:
        if col in df.columns:
            df.drop(columns=[col], inplace=True)

    for col in feature_columns:
        if col not in df.columns:
            df[col] = 0.0
    return df[feature_columns]


def price_asteroids(
    asteroids: list[dict],
    capital: float,
    round_info: dict,
) -> list[float]:
    """
    Bid on a batch of asteroids.

    Pipeline:
      1. Predict catastrophe probs, outgassing impact, value, yield, delay.
      2. Compute P_safe = P(no catastrophe) * (1 - adjusted outgassing impact).
      3. Compute present-value expected profit.
      4. Apply winner's curse, economic cycle, cluster concentration, and
         capital management adjustments.
    """
    import numpy as np
    import pandas as pd

    n = len(asteroids)
    if n == 0:
        return []
    if capital <= CRITICAL_CAPITAL_THRESHOLD:
        return [0.0] * n

    feat_cols = _model["feature_columns"]

    # ---- build feature matrix ----
    df = pd.DataFrame(asteroids)
    cluster_ids = (
        df["cluster_id"].values if "cluster_id" in df.columns else np.full(n, -1)
    )
    X = _engineer_features(df, feat_cols)

    # ---- model predictions (5 models) ----
    cat_proba = _model["catastrophe_model"].predict(X)       # (n, 4)
    impact_proba = _model["outgassing_model"].predict(X)     # (n,) P(impact)
    pred_value = np.maximum(_model["value_model"].predict(X), 0.0)
    pred_yield = np.clip(_model["yield_model"].predict(X), 0.0, 1.5)
    pred_delay = np.clip(_model["delay_model"].predict(X), 1.0, 20.0)

    p_none = cat_proba[:, 0]
    p_void = cat_proba[:, 1]
    p_collapse = cat_proba[:, 2]
    p_outgas = cat_proba[:, 3]

    # ---- cluster bookkeeping ----
    cluster_members: dict[int, list[tuple[int, float]]] = {}
    cluster_counts: dict[int, int] = {}
    for i, cid in enumerate(cluster_ids):
        cluster_counts[cid] = cluster_counts.get(cid, 0) + 1
        cluster_members.setdefault(cid, []).append((i, p_outgas[i]))

    # ---- adjusted outgassing impact (blend model + cluster-peer risk) ----
    adjusted_impact = impact_proba.copy()
    for cid, members in cluster_members.items():
        if len(members) <= 1:
            continue
        for idx, _ in members:
            # v2's probabilistic cluster math: P(any peer outgasses)
            peer_probs = [p for j, p in members if j != idx]
            p_no_peer_outgas = 1.0
            for p in peer_probs:
                p_no_peer_outgas *= 1.0 - p
            peer_risk = 1.0 - p_no_peer_outgas
            # v1's blending: take max of model prediction and peer-derived risk
            adjusted_impact[idx] = max(adjusted_impact[idx], peer_risk * 0.5)

    # ---- expected value (v1's formula with dedicated outgassing model) ----
    p_safe = p_none * (1.0 - adjusted_impact)

    expected_revenue = p_safe * pred_value * pred_yield

    expected_penalty = p_void * 100 + p_collapse * 200 + p_outgas * 300
    for i, cid in enumerate(cluster_ids):
        if cluster_counts[cid] > 1:
            expected_penalty[i] += p_outgas[i] * 10 * (cluster_counts[cid] - 1)

    expected_profit = expected_revenue - expected_penalty

    # Time-value discounting
    risk_free_rate = round_info.get("risk_free_rate", 0.005)
    expected_profit *= 1.0 / (1.0 + risk_free_rate) ** pred_delay

    # ---- winner's curse (v1's parameterisation) ----
    num_competitors = round_info.get("num_active_competitors", 5)
    wc_factor = max(
        0.25,
        WINNERS_CURSE_BASE
        - WINNERS_CURSE_PER_COMPETITOR * max(0, num_competitors - 3),
    )

    bids = np.maximum(0.0, expected_profit * wc_factor)

    # ---- per-asteroid cap (v1: 12%) ----
    bids = np.minimum(bids, capital * MAX_BID_FRACTION)

    # ---- pending-extraction throttle (v1's 3 tiers) ----
    pending = round_info.get("num_pending_extractions", 0)
    if pending > 6:
        bids *= 0.65
    elif pending > 4:
        bids *= 0.80
    elif pending > 2:
        bids *= 0.90

    # ---- late-game aggressiveness (v1) ----
    total_rounds = round_info.get("total_rounds", 50)
    round_number = round_info.get("round_number", 1)
    rounds_left = total_rounds - round_number

    if rounds_left <= 3:
        pending_revenue = round_info.get("pending_revenue", 0)
        if pending_revenue > capital * 0.3:
            bids *= 1.4
    elif rounds_left <= 8:
        bids *= 1.1

    # ---- economic cycle adjustment (v1) ----
    econ = asteroids[0].get("economic_cycle_indicator", 1.0) if asteroids else 1.0
    if econ > 1.2:
        bids *= 1.05
    elif econ < 0.8:
        bids *= 0.90

    # ---- low-capital safety (v1) ----
    if capital < LOW_CAPITAL_THRESHOLD:
        bids *= 0.3

    # ---- cluster concentration (v1: best per cluster, penalise others) ----
    for cid, members in cluster_members.items():
        if len(members) <= 1:
            continue
        best_idx = max(members, key=lambda x: bids[x[0]])[0]
        for idx, _ in members:
            if idx != best_idx:
                bids[idx] *= CLUSTER_PENALTY_FACTOR

    # ---- portfolio-level cap (v1: 70%) ----
    total = bids.sum()
    max_total = capital * MAX_PORTFOLIO_FRACTION
    if total > max_total:
        bids *= max_total / total

    # ---- minimum bid filter ----
    bids[bids < MIN_BID] = 0.0

    return bids.tolist()
