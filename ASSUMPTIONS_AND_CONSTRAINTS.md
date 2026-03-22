# Assumptions, Limitations & Constraints

Derived from: README.md, SUBMISSION_GUIDE.md, DATA_DICTIONARY.md, and example_strategy.py.

---

## Hard Constraints (Non-Negotiable Rules)

### Submission Format
- Single Python file named `strategy.py` with a `price_asteroids()` function.
- Optionally one pre-trained model file (max 50 MB), loadable offline.
- Allowed formats: `.joblib`, `.pkl`, `.pt`, `.pth`, `.safetensors`, `.json`.

### Runtime Environment
- Python 3.11 in an isolated sandbox with **no network access** and **no arbitrary filesystem access**.
- `load_model()` timeout: **30 seconds**.
- `price_asteroids()` timeout: **2 seconds per round**.
- Pinned package versions — must train with exact versions:
  - numpy 1.26.4, pandas 2.2.0, scipy 1.12.0, scikit-learn 1.4.0
  - xgboost 2.0.3, lightgbm 4.3.0, catboost 1.2.2, statsmodels 0.14.1
  - torch 2.2.0 (CPU only), joblib 1.3.2
- No other packages available at competition time.

### Bidding Mechanics
- **First-price sealed-bid auction**: highest bid wins and pays their bid.
- If total non-zero bids exceed capital, **all bids are scaled down proportionally**.
- Bid 0 to pass on an asteroid.
- One function call per round receives the **entire batch** of asteroids simultaneously.

### Capital & Bankruptcy
- Capital is depleted immediately when you win an auction (bid paid upfront).
- Revenue arrives after a variable `extraction_delay` (in rounds).
- Liquid capital earns a per-round `risk_free_rate` (compound interest baseline to beat).
- **Bankruptcy at capital <= 0 is permanent elimination.**
- Revenue from extractions still in progress is collected at sector end, even if delay exceeds final round.

### Statefulness
- The model object returned by `load_model()` is **deep-copied before each round** — no state persists between rounds via the model.
- No module-level mutable state should be relied upon (unclear if enforced, but documented as intended).

---

## Key Assumptions (Inferred from Documentation)

### About the Data
1. **Training data may not match competition distribution.** The DATA_DICTIONARY explicitly warns: "Market conditions and data collection methods may differ between the training dataset and live competition sectors." Models must generalize.
2. **Training data spans multiple time periods** (the `time_period` column), but `time_period` must NOT be used as a model feature.
3. **Catastrophe/impacted rows have zeroed targets.** Rows where `catastrophe_type != "none"` OR `toxic_outgassing_impact == 1` have `mineral_value = 0` and `extraction_yield = 0`. These must be **filtered before training regression models** for value/yield, but **included when training catastrophe classifiers**.
4. **~95 features available**, but not all are informative. Feature selection/importance analysis is critical.
5. **Some features are intentionally noisy or misleading:**
   - `lucky_number` — "Numerological favorability score" (likely noise/trap feature).
   - `media_hype_score` — "May not correlate with actual value."
   - `social_sentiment_score` — "Notoriously noisy."
   - `ai_valuation_estimate` — "Use with caution."
   - `analyst_consensus_estimate` — "Recent investigations have raised questions about analyst independence."
   - `surveyor_reputation` — "Self-reported and updated annually."
6. **Feature interactions matter.** "Simple univariate relationships may not capture the full picture." Tree-based models or explicit interaction features are favored.
7. **Economic cycle indicator** (0.7=bust, 1.0=normal, 1.4=boom) is consistent for all asteroids in a round. This is a key regime variable that affects all valuations.

### About the Auction
8. **5 competitors per group** in preliminary rounds; ~8 in finals. Bidding strategy must account for competitive dynamics.
9. **Winner's curse is real.** In a first-price sealed-bid auction, the winner is the bidder who overestimated the most. Bids must be discounted below expected value.
10. **No information about other teams' strategies.** Only `previous_round` results (winning bids, catastrophes) and `market_history` provide competitive intelligence.
11. **Scoring is average final capital across multiple group assignments.** Consistency is rewarded over high-variance strategies.

### About Catastrophes
12. **Catastrophe probability is feature-driven**, primarily by: structural_integrity (primary), density, porosity, volatile_content, and survey_confidence.
13. **Toxic outgassing has cluster-wide spillover.** If ANY asteroid in a cluster outgasses, ALL other asteroids in that same cluster have yield zeroed. This means cluster_id is a critical risk factor.
14. **Catastrophe penalties are additive to losing the bid:**
    - Void Rock: -$100
    - Structural Collapse: -$200
    - Toxic Outgassing: -$300 - $10 × (other asteroids in cluster)

### About Extraction
15. **Revenue = mineral_value × extraction_yield**, where both must be predicted.
16. **Extraction yield is influenced by** equipment_compatibility, survey data quality, and surface environment (not a simple constant).
17. **Extraction delay depends on** difficulty, location, accessibility, and size — this is a cash-flow timing concern, not just a valuation concern.

---

## Strategic Limitations

### Information Gaps
- We don't know the competition asteroid distribution — only the 10,000 training samples.
- We don't know competitor strategies or bidding behavior in advance.
- We don't know exactly how many rounds or how many asteroids per round in each sector.
- We don't know the starting capital (the example uses $10,000 but this may differ).
- `mineral_value` and `extraction_yield` are only available in training — must be estimated from features during competition.

### Modeling Challenges
- **Multi-target prediction problem**: need to predict mineral_value, extraction_yield, extraction_delay, AND catastrophe probability.
- **Zeroed targets** create a mixture distribution — need separate models or a two-stage approach (predict catastrophe first, then value conditional on no catastrophe).
- **Cluster-level risk**: toxic outgassing impacts other asteroids in the same cluster, but we can't control which other asteroids competitors win in the same cluster.
- **Extraction delay creates cash-flow risk**: winning too many long-delay asteroids can starve capital.
- **Proportional scaling of bids**: if we bid on too many asteroids, all bids shrink, potentially losing everything. Must be deliberate about portfolio allocation.

### Operational Constraints
- 2-second timeout means heavy inference (large neural nets, ensemble of many models) may be risky.
- 50 MB model file limit constrains model complexity.
- CPU-only PyTorch — no GPU acceleration.
- Single model file — can't ship a folder of model artifacts (but can pack multiple models into one joblib/pickle file).

---

## Competition Structure Implications

### Preliminary Rounds
- Random groups of 5, multiple group appearances.
- Score = average final capital across all appearances.
- **Implication**: Strategy must be robust across different opponent mixes, not tuned to beat one specific rival.

### Finals
- Top teams head-to-head, multiple runs across sectors.
- Score = average capital across all final runs.
- **Implication**: Consistent strategies beat volatile ones.

### Three Sectors
- Outer Rim, Inner Belt, Core Belt — each with different economic conditions.
- The same strategy file is used across all sectors.
- **Implication**: Strategy must adapt dynamically using `round_info` and asteroid features (especially `economic_cycle_indicator`, `belt_region`).

---

## Red Flags / Trap Features to Investigate

| Feature | Concern |
|---------|---------|
| `lucky_number` | Almost certainly noise. |
| `media_hype_score` | Explicitly warned as unreliable. |
| `social_sentiment_score` | "Notoriously noisy." |
| `ai_valuation_estimate` | "Use with caution." Could be miscalibrated or adversarial. |
| `analyst_consensus_estimate` | Independence questioned. May contain bias. |
| `surveyor_reputation` | Self-reported. May be inflated. |
| `infrastructure_proximity` | "Measured at time of survey; infrastructure expands over time." Stale data. |
| `time_period` | Explicitly forbidden as a model feature. |

---

## Summary of What Must Be Predicted

| Target | Type | Notes |
|--------|------|-------|
| `mineral_value` | Regression | Zeroed for catastrophe/impacted rows. Filter these when training. |
| `extraction_yield` | Regression (0-1+) | Same zeroing caveat. Determines actual revenue fraction. |
| `extraction_delay` | Regression/Classification (integer rounds) | Affects cash flow timing. |
| `catastrophe_type` | Multi-class classification | none / void_rock / structural_collapse / toxic_outgassing |
| `toxic_outgassing_impact` | Binary classification | Cluster-level spillover risk |

**Net expected value per asteroid** ≈ P(no catastrophe) × P(no outgassing impact) × E[mineral_value] × E[extraction_yield] - P(catastrophe) × E[penalty] - bid

The bid must be **below** this expected value to be profitable, and **further discounted** for winner's curse and cash-flow risk from extraction delay.