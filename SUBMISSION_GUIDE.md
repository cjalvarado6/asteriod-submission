# Submission Guide

## Function Signature

Create a single Python file with your strategy. You can use either the **simple interface** (one function) or the **ML interface** (two functions for heavy model loading).

### Simple Interface

For strategies without heavy ML models:

```python
# my_strategy.py

STRATEGY_NAME = "Your Corp Name"  # Optional — defaults to filename

def price_asteroids(asteroids: list[dict], capital: float, round_info: dict) -> list[float]:
    """
    Called once per round with the full batch of asteroids.

    Args:
        asteroids: list of feature dicts (~95 values each, mostly float,
                   some categorical strings like spectral_class, belt_region, probe_type).
        capital: your current liquid capital (what you can bid with right now)
        round_info: dict with keys:
            - round_number: current round (1-indexed)
            - total_rounds: total rounds in this sector
            - sector_name: name of the current sector
            - asteroids_this_round: number of asteroids this round
            - risk_free_rate: per-round interest rate on liquid capital
            - num_active_competitors: number of non-bankrupt competitors
            - pending_revenue: your total expected revenue from in-progress extractions
            - num_pending_extractions: number of your extractions still in progress
            - previous_round: list of per-asteroid results from last round (None for round 1)
                Each entry: {winning_bid, was_sold, was_catastrophe, you_won}
            - market_history: cumulative market stats (None for round 1)
                Keys: rounds_completed, cumulative_asteroids_offered,
                      cumulative_asteroids_sold, cumulative_catastrophes,
                      avg_winning_bid_last5, your_total_wins, your_total_spending
        
        Note: Economic cycle is in each asteroid's features as economic_cycle_indicator
              (0.7=bust, 1.0=normal, 1.4=boom), consistent for all asteroids in a round.

    Returns:
        List of bid amounts (same length as asteroids). Return 0 to pass on an asteroid.
        If total bids exceed capital, all bids are scaled down proportionally.
    """
    return [0.0] * len(asteroids)
```

### ML Interface (Recommended for Heavy Models)

For strategies using sklearn, PyTorch, XGBoost, or other ML libraries with slow import/load times, use the **two-function interface**:

```python
# my_ml_strategy.py

STRATEGY_NAME = "ML Corp"

def load_model():
    """
    Called ONCE at tournament start with 30-second timeout.
    Use this to import heavy libraries and load your trained model.
    
    Returns:
        Any object (dict, model, etc.) that will be passed to price_asteroids.
    """
    import joblib
    import os
    model_dir = os.path.dirname(os.path.abspath(__file__))
    return joblib.load(os.path.join(model_dir, "my_model.joblib"))


def price_asteroids(asteroids: list[dict], capital: float, round_info: dict, model=None) -> list[float]:
    """
    Called each round with 2-second timeout.
    
    Args:
        asteroids, capital, round_info: Same as simple interface
        model: The object returned by load_model() (or None if load_model not defined)
    
    Returns:
        List of bid amounts (same length as asteroids).
    """
    if model is None:
        return [0.0] * len(asteroids)  # Fallback if model failed to load
    
    # Use your pre-loaded model for fast predictions
    predictions = model.predict(...)
    return compute_bids(predictions, capital)
```

**Key points for ML interface:**
- `load_model()` is optional but recommended for ML strategies
- `load_model()` has a **30-second timeout** for importing libraries and loading models
- `price_asteroids()` has a **2-second timeout** per round — keep it fast!
- The model object is **deep-copied** before each round to prevent state storage
- Do NOT store state in the model object between rounds — it will be reset
- Heavy imports (torch, sklearn, xgboost) should happen inside `load_model()`, not at module level

## Portfolio Bidding

You see all asteroids in a round at once and submit bids on the entire batch simultaneously. This means you can:
- Compare asteroids against each other and bid on the most attractive ones
- Allocate your capital budget across the batch
- Consider how many asteroids you want to win in a single round

If the sum of your non-zero bids exceeds your available capital, all bids are scaled down proportionally to fit within your budget.

## Market Intelligence

Starting from round 2, `round_info` includes:

- **`previous_round`**: A list with one entry per asteroid from last round. Each entry tells you the winning bid (or `None` if unsold), whether a catastrophe occurred, and whether you won it.

- **`market_history`**: Cumulative stats including total asteroids sold, catastrophes observed, recent average winning bids, and your own win/spending totals.

## Environment Setup

Before developing your strategy, set up your environment with the allowed packages.

### Option 1: Conda (Recommended)

```bash
conda env create -f environment.yml
conda activate asteroid-competition
```

This installs all allowed packages including XGBoost, LightGBM, CatBoost, PyTorch, and more.

### Option 2: pip

Install core dependencies:
```bash
pip install -e .
```

Or install with all ML packages:
```bash
pip install -e ".[all]"
```

See `README.md` for the complete list of allowed packages and versions.

---

## Test Your Strategy

Use the training data (`data/training.parquet`) to develop and validate your model. The training data includes target variables not available during competition: `mineral_value`, `extraction_yield`, `extraction_delay`, `catastrophe_type`, and `toxic_outgassing_impact`.

**Important**: Rows with `catastrophe_type != "none"` or `toxic_outgassing_impact == 1` have zeroed `mineral_value` and `extraction_yield`.

```python
import pandas as pd

df = pd.read_parquet("data/training.parquet")

# Build a batch of asteroid feature dicts (drop target columns)
target_cols = ["mineral_value", "extraction_yield", "extraction_delay", "catastrophe_type", "toxic_outgassing_impact"]
batch = []
for _, row in df.head(10).iterrows():
    features = row.drop(target_cols).to_dict()
    features.pop("asteroid_id", None)
    batch.append(features)

# Simulate a round
bids = price_asteroids(batch, capital=10000.0, round_info={
    "round_number": 1,
    "total_rounds": 100,
    "sector_name": "Outer Rim",
    "asteroids_this_round": 10,
    "risk_free_rate": 0.002,
    "num_active_competitors": 5,
    "pending_revenue": 0.0,
    "num_pending_extractions": 0,
    "previous_round": None,
    "market_history": None,
})

for i, bid in enumerate(bids):
    mineral_val = df.iloc[i]["mineral_value"]
    print(f"Asteroid {i}: bid={bid:.2f}, mineral_value={mineral_val:.2f}")
```

## Submission Format

Your submission is a **directory** containing:

```
my_submission/
├── strategy.py      # Required: your bidding strategy
└── model.joblib     # Optional: one pre-trained model file
```

### Strategy File (Required)

A single Python file named `strategy.py` with your `price_asteroids` function.

### Model File (Optional)

You may include **one** pre-trained model file. Supported formats:

| Format | Extension | Framework |
|--------|-----------|----------|
| Joblib | `.joblib` | scikit-learn |
| Pickle | `.pkl` | scikit-learn, general |
| PyTorch | `.pt`, `.pth` | PyTorch |
| SafeTensors | `.safetensors` | Any (recommended for large models) |
| JSON | `.json` | XGBoost, LightGBM, custom |

**Constraints:**
- Maximum file size: **50 MB**
- Only one model file allowed
- Model must be loadable without network access

### Loading Your Model

Use `__file__` to locate your model relative to your strategy:

```python
import os
import joblib

# Load model at module level (runs once)
_model_path = os.path.join(os.path.dirname(__file__), "model.joblib")
_model = joblib.load(_model_path)

def price_asteroids(asteroids, capital, round_info):
    # Use _model for predictions
    ...
```

---

## Rules

1. **One strategy file + one optional model file.** Your submission directory contains `strategy.py` and optionally one model file.
2. **No network access.** Strategies run in an isolated sandbox with no internet.
3. **2-second timeout.** Your function must return within 2 seconds per round.
4. **No arbitrary filesystem access.** You can only read your own model file.
5. **Allowed packages:** numpy, pandas, scipy, scikit-learn, xgboost, lightgbm, catboost, statsmodels, torch (CPU), joblib. **Versions are pinned exactly** — see `README.md` for the version table. Using different versions may cause model loading failures.

## What You Have

- **Training data**: `data/training.parquet` with ~95 features + target values for 10,000 asteroids
- **Feature reference**: `DATA_DICTIONARY.md` with descriptions of every feature
- **Example strategy**: `strategies/example_strategy.py` — a simple heuristic bidder

## What You Don't Have

- The mineral value, extraction yield, or extraction delay during competition (you must estimate these from the ~95 measurement features)
- Other teams' strategies
- Advance knowledge of which specific asteroids will appear in competition

## Tournament Structure

The competition runs in two phases:

### 1. Preliminary Rounds

- All teams are randomly assigned to **groups of 5**
- Each team participates in **multiple different groups**
- Each group plays through one selected sector (Outer Rim, Inner Belt, or Core Belt)
- Your score is your **average final capital** across all group appearances
- The **top teams** by average score advance to the finals

### 2. Finals

- All finalists compete together
- The finals are run multiple times across different sectors
- Your final score is your **average capital** across all runs
- The team with the highest average wins

## How You Win

You compete against other participants in randomized groups. Your preliminary ranking is based on **average performance across many different opponent combinations**. This rewards consistent strategies over lucky matchups.

In the finals, the **8 best teams** compete head-to-head across multiple runs. The winner is determined by **average final capital** across all final runs — not a single elimination bracket.

Asteroids won in later rounds still count. Revenue is collected at sector end even if extraction delay extends past the final round.
