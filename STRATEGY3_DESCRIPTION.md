# Strategy v3 — Deep Rock Mining

## Overview

Strategy v3 is a five-model LightGBM ensemble paired with a risk-aware bidding engine. It combines the best elements of two earlier iterations: v2's superior feature engineering (mineral-price interaction terms, engineered composites) with v1's richer model architecture (dedicated outgassing impact classifier) and battle-tested capital management logic.

The core insight is that profitable bidding in a first-price sealed-bid auction requires solving two distinct problems simultaneously:

1. **Valuation** — estimate the true expected value of each asteroid.
2. **Bid shading** — determine how far below your valuation to bid, accounting for the winner's curse, cash-flow risk, and competitive dynamics.

---

## Model Architecture

Five LightGBM models are trained offline and bundled into a single `.joblib` file:

| # | Model | Target | Training Set | Task |
|---|-------|--------|-------------|------|
| 1 | Catastrophe classifier | `catastrophe_type` (4 classes) | All rows | Multiclass classification |
| 2 | Outgassing impact classifier | `toxic_outgassing_impact` (binary) | All rows | Binary classification |
| 3 | Mineral value regressor | `mineral_value` | Clean rows only | Regression |
| 4 | Extraction yield regressor | `extraction_yield` | Clean rows only | Regression |
| 5 | Extraction delay regressor | `extraction_delay` | All rows | Regression |

### Why five models instead of one?

Asteroid outcomes follow a **mixture distribution**. Roughly 15–20% of asteroids suffer a catastrophe or toxic outgassing impact, which zeroes their mineral value and extraction yield. Training a single regressor on the full dataset would blend catastrophe-zero rows with healthy rows, biasing predictions downward for good asteroids and upward for bad ones.

The two-stage approach mirrors the data-generating process:

```
Nature decides:  catastrophe? ──yes──▶ value = 0, yield = 0, you pay a penalty
                      │
                     no
                      │
                      ▼
                 outgassing impact? ──yes──▶ yield = 0
                      │
                     no
                      ▼
                 mineral_value ~ f(features)
                 extraction_yield ~ g(features)
```

By filtering catastrophe/impacted rows before training the value and yield regressors, those models learn the conditional distribution `E[value | safe]` and `E[yield | safe]`, which are then combined with the safety probabilities at inference time.

The outgassing impact model (model 2) is separate from the catastrophe classifier (model 1) because outgassing impact is a **cluster-level spillover effect** — it happens to an asteroid not because of its own properties, but because a neighbour in its geological cluster outgassed. This requires its own classifier.

---

## Feature Engineering

### Dropped features

| Feature | Reason |
|---------|--------|
| `time_period` | Explicitly forbidden by competition rules |
| `asteroid_id` | Identifier, no predictive value |
| `lucky_number` | Documented as "numerological favorability score" — noise |
| `media_hype_score` | DATA_DICTIONARY warns "may not correlate with actual value" |
| `social_sentiment_score` | DATA_DICTIONARY warns "notoriously noisy" |
| `cluster_id` | Arbitrary identifier with no ordinal meaning; used strategically in the bidding logic instead |
| `ai_valuation_estimate` | DATA_DICTIONARY warns "Use with caution" — high distribution-shift risk between training and competition |
| `analyst_consensus_estimate` | DATA_DICTIONARY warns "questions about analyst independence" — unreliable and shift-prone |

### Engineered features

| Feature | Formula | Rationale |
|---------|---------|-----------|
| `mineral_val_{metal}` | `mineral_signature_{metal} × mineral_price_{metal}` | Captures the dollar-value of each mineral component. The interaction between concentration and spot price is the fundamental driver of `mineral_value`. |
| `mineral_val_water` | `water_ice_fraction × mineral_price_water` | Same logic for water ice. |
| `total_mineral_estimate` | Sum of all `mineral_val_*` columns | Provides a single aggregate value estimate for the model to anchor on. |
| `risk_score` | `(1 - structural_integrity) × porosity × (1 + volatile_content)` | Composite catastrophe risk indicator. Combines the three documented primary risk factors into a single feature the tree model can split on directly. |
| `access_difficulty_ratio` | `accessibility_score / (extraction_difficulty + 0.01)` | Ratio of how reachable the deposits are vs. how hard they are to extract. A high ratio means easy, profitable extraction. |
| `survey_quality` | `survey_confidence × data_completeness × (1 - conflicting_results × 0.3)` | Composite data reliability score. Low-quality surveys mean higher uncertainty in all predictions. |

### Categorical encoding

Categorical features (`spectral_class`, `belt_region`, `probe_type`) are mapped to integer codes and declared as categorical to LightGBM, which handles them with its native optimal-split algorithm rather than treating them as ordinal numbers.

---

## Bidding Logic — Mathematical Framework

### Step 1: Expected value estimation

For each asteroid *i*, the expected gross value is:

$$
\text{EV}_i = P(\text{safe})_i \times \hat{V}_i \times \hat{Y}_i - \mathbb{E}[\text{penalty}]_i
$$

Where:

- $P(\text{safe})_i = P(\text{no catastrophe})_i \times (1 - P(\text{impact})_i^{\text{adj}})$
- $\hat{V}_i$ = predicted mineral value (from model 3)
- $\hat{Y}_i$ = predicted extraction yield (from model 4)
- $\mathbb{E}[\text{penalty}]_i = P(\text{void})_i \times 100 + P(\text{collapse})_i \times 200 + P(\text{outgas})_i \times (300 + 10 \times N_{\text{cluster}})$

This formula directly mirrors the payout structure: you earn revenue only if the asteroid is safe (no catastrophe AND no outgassing impact), and you pay a probabilistically-weighted expected penalty otherwise.

### Step 2: Outgassing impact adjustment

The raw outgassing impact probability from model 2 is adjusted using **cluster-level peer risk**. For asteroid *i* in a cluster with peers *j₁, j₂, …, jₖ*:

$$
P(\text{any peer outgasses}) = 1 - \prod_{j \neq i} \bigl(1 - P(\text{outgas})_j\bigr)
$$

This is a standard inclusion-exclusion probability for independent events. If any peer in the cluster outgasses, asteroid *i*'s yield drops to zero. The final adjusted impact probability blends the model's direct prediction with this peer-derived risk:

$$
P(\text{impact})_i^{\text{adj}} = \max\bigl(P(\text{impact})_i^{\text{model}},\; 0.5 \times P(\text{any peer outgasses})\bigr)
$$

The 0.5 coefficient reflects that not all peers visible in the current round represent the full cluster — some cluster members may appear in other rounds or not at all.

### Step 3: Time-value discounting

Revenue from asteroid *i* arrives after a predicted delay of $\hat{D}_i$ rounds. Meanwhile, capital held in cash earns the risk-free rate $r$ each round. The present value of future revenue is discounted accordingly:

$$
\text{PV}_i = \frac{\text{EV}_i}{(1 + r)^{\hat{D}_i}}
$$

This ensures the strategy correctly penalises long-delay asteroids. An asteroid worth \$500 arriving in 10 rounds at $r = 0.5\%$ is worth $500 / 1.005^{10} \approx \$475.6$ in present-value terms. Meanwhile, the capital used to buy it could have compounded passively.

### Step 4: Winner's curse discount

In a first-price sealed-bid auction with $k$ competitors, the winner is the bidder whose estimate is highest — which in expectation means the most overestimated. The optimal bid in a symmetric independent private values (IPV) auction shades below the true value:

$$
b_i = \text{PV}_i \times w(k)
$$

where $w(k) = \max\bigl(0.25,\; 0.50 - 0.03 \times (k - 3)\bigr)$

The base discount of 50% (bid half of expected value) is aggressive, reflecting that asteroid valuations are inherently noisy. The per-competitor adjustment of 3% per additional competitor beyond 3 further shades the bid as competition intensifies, since more competitors increase the likelihood that the winning bid exceeds the true value.

### Step 5: Capital management

Several layers of bid scaling protect against bankruptcy:

| Condition | Action | Rationale |
|-----------|--------|-----------|
| Per-asteroid cap | $b_i \leq 12\%$ of capital | Limits concentration risk — no single bad asteroid can cause catastrophic loss. |
| Pending extractions > 6 | Bids × 0.65 | Capital is locked in transit; reduce exposure to avoid cash starvation. |
| Pending extractions > 4 | Bids × 0.80 | Moderate throttle. |
| Pending extractions > 2 | Bids × 0.90 | Light throttle. |
| Capital < \$100 | Bids × 0.30 | Survival mode — preserve remaining capital. |
| Capital ≤ \$50 | All bids = 0 | Full shutdown to avoid bankruptcy. |
| Portfolio total | Capped at 70% of capital | Always retain a 30% cash reserve for future rounds and interest compounding. |
| Minimum bid filter | Bids < \$15 → 0 | Don't waste capital on low-conviction bids that are unlikely to win. |

### Step 6: Economic cycle adjustment

The `economic_cycle_indicator` (0.7 = bust, 1.0 = normal, 1.4 = boom) is consistent for all asteroids in a round. The strategy adjusts bids:

- **Boom** (> 1.2): bids × 1.05 — mineral prices are elevated, actual revenues likely higher than historical training data suggests.
- **Bust** (< 0.8): bids × 0.90 — mineral prices are depressed, be more conservative.

This is a simple regime adjustment rather than a full recalibration, because the model already sees the mineral price features that partially capture cycle effects.

### Step 7: Cluster concentration control

For each geological cluster with multiple asteroids in the current round, the strategy keeps only the **highest-value bid** at full strength and scales down all others by 75%:

$$
b_j \leftarrow 0.25 \times b_j \quad \forall j \in \text{cluster},\; j \neq \arg\max_j b_j
$$

This mitigates toxic outgassing cluster risk. If you win multiple asteroids in the same cluster and one outgasses, you lose revenue on all of them. By concentrating bids on the single best asteroid per cluster, you limit correlated losses.

### Step 8: Late-game aggressiveness

In the final rounds (≤ 3 remaining), if there is substantial pending revenue (> 30% of current capital), bids are scaled up by 40%. In the last 8 rounds, a milder 10% increase is applied. The rationale: with fewer rounds remaining, the opportunity cost of holding cash drops (fewer rounds to compound), and pending revenue will arrive at sector end regardless of remaining round count.

---

## Training Methodology

- **Cross-validation**: 5-fold stratified CV (for classifiers) or standard KFold (for regressors) with early stopping.
- **Final model**: Trained on all data using the median best iteration from CV folds — avoids overfitting while using the full dataset.
- **Regularisation**: L1 (`reg_alpha = 0.1`) and L2 (`reg_lambda` varies by model) regularisation, feature subsampling (80%), and row subsampling (80%) per boosting round.
- **LightGBM**: Chosen for fast inference (critical for the 2-second timeout) and native categorical feature support.

---

## Design Decisions and Trade-offs

| Decision | Rationale |
|----------|-----------|
| Drop `ai_valuation_estimate` and `analyst_consensus_estimate` | Data dictionary flags both as unreliable ("Use with caution"; "questions about analyst independence"). High risk of distribution shift between training and competition — if the valuation methodology or analyst pool changes, models that depend on these features will degrade silently. Prefer features grounded in physical measurements. |
| Drop `cluster_id` from model, use in bidding logic | Cluster ID is an arbitrary integer with no ordinal meaning — useless as a regression/classification feature. But it is essential for computing cluster-level outgassing spillover risk in the bidding engine. |
| 70% portfolio cap (30% cash reserve) | Balances opportunity cost of idle capital against bankruptcy risk. The risk-free rate means idle capital is not truly idle. |
| Winner's curse base of 50% | Aggressive shading reflects high uncertainty in asteroid valuations. Empirically, overbidding is the primary cause of losses in first-price auctions with noisy estimates. |
| Separate outgassing impact model | Unlike direct catastrophes, outgassing impact is driven by cluster neighbourhood effects, not the asteroid's own geology. A dedicated model can learn this distinct signal. |
