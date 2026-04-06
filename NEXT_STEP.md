# Next Steps for Book Sales Forecasting

## Where We Are

**Best results (3yr panel, 30M rows, temporal sampling, LightGBM):**

| Horizon | WAPE (test) | Baseline WAPE | Improvement |
|---------|-------------|---------------|-------------|
| 15d | 0.652 | 0.792 | 17.6% |
| 30d | 0.604 | 0.773 | 21.8% |

---

## What the Data Tells Us

### The problem is extremely concentrated

- **Top 1% of items** (2,748) produce **61% of total sales volume** and **61% of total error**
- **Top 5%** (13,740) produce **84% of volume** and **80% of error**
- The remaining 95% of items are nearly zero-sellers in any 15d window

### The target is zero-inflated and heavy-tailed

- **76.5%** of test rows for 15d have target = 0
- Median target = 0, P90 = 3, P99 = 44, max = 85,235
- Raw target skewness = 195 (extreme); `log1p(target)` reduces skewness from 195 to 1.9

### The model systematically under-predicts high-volume items

| Target bucket | Rows | WAPE | Signed error |
|--------------|------|------|-------------|
| Zero | 76.5% | n/a (good: MAE=0.09) | +0.09 |
| 1 | 9.2% | 0.985 (near random) | - |
| 2-5 | 8.4% | 0.738 | - |
| 6-20 | 3.9% | 0.583 | - |
| 21-100 | 1.5% | 0.540 | - |
| 100+ | 0.4% (but 50% of volume) | 0.606 | **-226** |

The model is biased low for items with targets > 100. Mean signed error = -226 for the 100+ bucket. These are the items that matter most for inventory.

### Autocorrelation is weak

Even for the highest-volume items, daily autocorrelation is very low (0.05-0.12 at lag 1). This is a **sparse event** problem, not a smooth time series. The data behaves more like intermittent demand (Poisson-like with overdispersion) than a continuous signal.

### Strong seasonality exists but is captured weakly

- Peak months: July (week 27), September (week 35-36), February
- Trough: March-April (weeks 13-16)
- This aligns with Chinese academic cycles (new semester starts in Feb/Sep, summer stock-up in Jul)
- Current calendar features (`month`, `week_of_year`, `quarter`) have moderate importance but don't interact with item-level patterns

---

## Priority 1: High-Impact LightGBM Improvements

These stay within the current framework and can be tested quickly.

### 1A. Log-transform the target

**Why:** The raw target (skew=195) causes MAE/L1 loss to be dominated by a handful of extreme items. `log1p(target)` reduces skew to 1.9, letting the model learn patterns across the full volume range. At inference, `expm1(prediction)` recovers the original scale.

**Implementation:**
- In `_build_chunk_features`: `target_Nd = np.log1p(target_Nd)`
- In eval: `prediction = np.expm1(booster.predict(...))`
- Keep the original target column for metric computation

**Expected impact:** High. This single change typically improves WAPE by 5-15% in zero-inflated retail forecasting, because it stops the optimizer from spending all its budget on outliers.

### 1B. Two-stage model: classifier + regressor

**Why:** 76.5% of rows have target=0. The model currently spends most of its splits distinguishing "zero or nonzero?" rather than "how much when nonzero?". Separating these two decisions improves both.

**Implementation:**
- Stage 1: LightGBM binary classifier (`objective=binary`) predicting P(target > 0)
- Stage 2: LightGBM regressor (current model) trained only on rows where target > 0
- Final prediction: `P(sell) * E[qty | sell]`

**Expected impact:** High. The classifier can use different features (e.g., `days_since_sale` is extremely predictive of "will it sell at all?"), while the regressor focuses on quantity.

### 1C. Sample weights for high-volume items

**Why:** 0.4% of rows (target > 100) carry 50% of total sales volume but the model treats all rows equally. This is why the model under-predicts heavy hitters.

**Implementation:**
- `sample_weight = np.log1p(target)` or `sample_weight = np.sqrt(target + 1)`
- Pass to `lgb.Dataset(..., weight=sample_weight)`

**Expected impact:** Medium-high. Directly addresses the systematic under-prediction for high-volume items.

### 1D. Fix P2: category rolling features on calendar days

**Why:** Still computing `category_roll_mean_28` on sale-event dates, not calendar dates. For sparse categories, the "28-day" window spans months.

**Implementation:** Reindex `cat_daily` to a full date range per category before computing rolling features.

**Expected impact:** Medium. Category features are in top 20 but not top 5.

---

## Priority 2: Feature Engineering

### 2A. Croston's method features (intermittent demand)

**Why:** The data is classic intermittent demand — infrequent, irregular sales with many zeros. Croston's method decomposes this into two components: demand interval (how often) and demand size (how much when it occurs). These are natural features for the model.

**Implementation:**
- `demand_interval_mean`: rolling average of days between non-zero sales
- `demand_size_mean`: rolling average of QTY when QTY > 0
- `demand_probability`: `1 / demand_interval_mean` (probability of selling on any given day)

### 2B. Promotional / event calendar

**Why:** Strong seasonality (week 27, 35-36) aligns with Chinese academic cycles. External features for semester starts, exam periods, major holidays (Spring Festival, National Day, 618/Double 11 shopping festivals) would let the model capture calendar effects that `week_of_year` alone misses.

**Implementation:** Add binary features for known event windows. Even a simple `is_semester_start_month` feature could help.

### 2C. Item lifecycle features

**Why:** New books have a "launch spike" pattern; backlist titles have steady-state demand. Currently `item_age_days` captures this weakly. Better features:
- `days_since_first_sale` (how established is this item?)
- `peak_7d_qty_ever` (what was the launch peak?)
- `qty_trend_90d` (slope of rolling mean — is it rising or declining?)

### 2D. Store count acceleration

**Why:** `store_count_28` is the #1 feature for 30d. Adding `store_count_7` and `store_count_14` would let the model detect distribution changes (new stores picking up an item, or stores dropping it).

---

## Priority 3: Advanced Models

### 3A. Temporal Fusion Transformer (TFT)

**Why:** TFT is specifically designed for multi-horizon forecasting with mixed-type inputs (static metadata, time-varying known, time-varying observed). It handles intermittent demand well and provides interpretable attention weights.

**Advantages over LightGBM:**
- Learns per-item temporal patterns (LightGBM shares all patterns globally)
- Attention mechanism can learn complex calendar interactions
- Native multi-horizon output (one forward pass gives all horizons)
- Provides prediction intervals

**Disadvantages:**
- Longer training time (GPU-intensive, but you have 4x A4000)
- More hyperparameter tuning
- Harder to debug

**Implementation:** Use PyTorch Forecasting library (`TemporalFusionTransformer`). Feed the same features, but structured as sequences:
- Static categoricals: ITEM_CATEORY_CODE, BPDNAME, UN_NUMBER, etc.
- Time-varying known: calendar features, item_age_days
- Time-varying observed: QTY, store_count, rolling features

**When to try:** After exhausting Priority 1 improvements. If LightGBM WAPE plateaus around 0.55-0.60, TFT could push to 0.45-0.50.

### 3B. N-HiTS (Neural Hierarchical Interpolation)

**Why:** State-of-art on M-competition benchmarks. Designed for multi-scale temporal patterns. Lighter than TFT, faster to train.

**When to try:** If TFT is too slow to iterate on. N-HiTS is a good middle ground between LightGBM speed and deep learning expressiveness.

### 3C. LightGBM + Neural Ensemble

**Why:** LightGBM excels at feature interactions and tabular patterns. Neural models excel at sequential patterns. Ensembling them often beats either alone.

**Implementation:**
- Train LightGBM as current (possibly with Priority 1 improvements)
- Train TFT or N-HiTS separately
- Ensemble: `final_pred = alpha * lgbm_pred + (1-alpha) * neural_pred`
- Optimize alpha on validation set (or learn it with a meta-model)

**When to try:** After both individual models are tuned.

---

## Priority 4: Evaluation & Business Alignment

### 4A. Per-item WAPE distribution

**Why:** Aggregate WAPE hides the distribution. An overall WAPE of 0.65 could mean "all items at 0.65" or "50% of items at 0.30 and 50% at 1.0". The business cares about different items differently.

### 4B. Stratified evaluation by item tier

Define item tiers by revenue/volume (A/B/C classification):
- Tier A (top 5% by volume): must have WAPE < 0.40
- Tier B (next 15%): WAPE < 0.60
- Tier C (remaining 80%): WAPE < 0.80 is acceptable

Track WAPE per tier. Optimize for Tier A first.

### 4C. Asymmetric loss

**Why:** In book retail, the cost of under-stocking (lost sales) may differ from over-stocking (carrying cost). If lost sales cost 3x carrying cost, the optimal forecast is biased upward.

**Implementation:** Custom LightGBM objective with asymmetric MAE:
```python
def asymmetric_mae(y_true, y_pred):
    residual = y_true - y_pred
    grad = np.where(residual > 0, -alpha, (1 - alpha))  # alpha > 0.5 penalizes under-prediction more
    hess = np.ones_like(y_true)
    return grad, hess
```

### 4D. Probabilistic forecasting (CRPS)

**Why:** Point forecasts don't capture uncertainty. For inventory decisions, knowing "50-150 units (90% CI)" is more useful than "100 units". LightGBM quantile regression (`objective=quantile`) can produce calibrated prediction intervals.

---

## Suggested Execution Order

| Step | What | Effort | Expected WAPE Impact |
|------|------|--------|---------------------|
| 1 | Log-transform target | 1 hour | -0.03 to -0.08 |
| 2 | Sample weights for high-volume items | 30 min | -0.02 to -0.05 |
| 3 | Two-stage classifier + regressor | 3 hours | -0.03 to -0.07 |
| 4 | Fix P2 (calendar-day rolling) | 1 hour | -0.01 to -0.02 |
| 5 | Croston's features + store count windows | 2 hours | -0.01 to -0.03 |
| 6 | Per-tier evaluation | 1 hour | (no WAPE change — but reveals where to focus) |
| 7 | TFT model | 1-2 days | -0.05 to -0.15 (uncertain) |
| 8 | LightGBM + TFT ensemble | 3 hours | -0.02 to -0.05 on top of best individual |

**Realistic target with Priority 1 alone: 15d WAPE ~0.55, 30d WAPE ~0.50**

**Realistic target with neural ensemble: 15d WAPE ~0.45-0.50, 30d WAPE ~0.40-0.45**
