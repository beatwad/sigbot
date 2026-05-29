# train_model.ipynb — Remaining Issues

Points 1, 2, 3 and 5 are already fixed.


## Point 4 — BTC.D / BTCDOM merge assigns back `"time"` column and may explode on duplicates

In `add_indicators` (cell 13):

```python
df[btcd_cols]   = pd.merge(df[["time"]], btcd[btcd_cols],   how="left", on="time")
df[btcdom_cols] = pd.merge(df[["time"]], btcdom[btcdom_cols], how="left", on="time")
```

`btcd_cols` and `btcdom_cols` both include `"time"`, so the merge result's `"time"` column is assigned back into `df["time"]` (harmless today, but confusing). More importantly, if `btcd` or `btcdom` contain duplicate timestamps, the left merge returns more rows than `df`, and the column assignment raises `ValueError`.

The same pattern appears in `inference.py` (`prepare_data`).

**Fix:**
```python
btcd_data_cols   = [c for c in btcd_cols   if c != "time"]
btcdom_data_cols = [c for c in btcdom_cols if c != "time"]

df[btcd_data_cols]   = pd.merge(df[["time"]], btcd[btcd_cols].drop_duplicates("time"),
                                how="left", on="time")[btcd_data_cols].values
df[btcdom_data_cols] = pd.merge(df[["time"]], btcdom[btcdom_cols].drop_duplicates("time"),
                                how="left", on="time")[btcdom_data_cols].values
```

Apply the same fix in `ml/inference.py` (`prepare_data`, lines 107–108).

---

Rows that hit TP early (target already set to 1) are retained even though their price history is incomplete. Rows that ran out of data with `target=0` are dropped. This asymmetry inflates the `target=1` class rate.

**Fix:** skip the row unconditionally when data ends early:
```python
if target_buy.shape[0] == 0 or target_sell.shape[0] == 0:
    pass_cycle = True
    break
```

---

## Point 6 — `buy_hours_to_save` / `sell_hours_to_save` are hardcoded magic lists

Cell 15 filters signal data to specific hours before building the dataset:

```python
buy_hours_to_save  = [0, 1, 3, 15, 16, 17, 18, 19, 21, 22, 23]
sell_hours_to_save = [2, 5, 8, 9, 11, 14, 17]
```

Cell 27 already derives statistically valid hours using confidence intervals on the class-1 ratio, but that result is never fed back to control this filter. The lists also overlap (hour 17 appears in both), which means the same signal time can enter both `train_buy` and `train_sell`, creating data leakage.

**Fix:** drive the hour lists from the CI-based analysis in cell 27 instead of hardcoding them, and assert `set(buy_hours_to_save).isdisjoint(sell_hours_to_save)`.

---

## Point 7 — No global random seed

Several places use non-deterministic randomness:
- `random.choices` (cell 9, ticker sampling for plots)
- `np.random.permutation` (inside `PumpDump.get_indicator` and `HighVolume.get_indicator`)

Results are non-reproducible between runs.

**Fix:** add `random.seed(42); np.random.seed(42)` near the top of the notebook (after imports) and pass `random_state=42` to any sklearn/optuna calls.

---

## Point 8 — `dropna()` called after `create_train_df` without investigation

```python
train_buy = create_train_df(...).dropna()
```

NaN sources (gaps in higher-TF merge, indicator failures returning 0, missing BTC.D coverage) are silently dropped. If the NaN rate changes between dataset refreshes, it is invisible.

**Fix:** log NaN counts per column before dropping:
```python
nan_counts = train_buy.isnull().sum()
if nan_counts.any():
    print("NaNs per column:\n", nan_counts[nan_counts > 0])
train_buy = train_buy.dropna()
```

---

## Point 9 — Indicator failures silently replaced with 0 (`except BaseException`)

In `indicators/indicators.py`, every indicator wraps its TA-Lib call in `except BaseException: value = 0`. A 0 RSI or 0 ATR looks like valid data to the model. Rows with silent indicator failures enter the training set contaminating feature distributions.

**Fix:** at minimum log the failure; better, propagate NaN (which `dropna()` would then catch):
```python
except Exception as e:
    logger.warning(f"Indicator {self.name} failed for {ticker}: {e}")
    rsi = np.nan  # or pd.Series(np.nan, index=df.index)
```

## Point 10 — `CFG.last_date` is hardcoded

```python
last_date = datetime.strptime("2024-11-20:18:00:00", "%Y-%m-%d:%H:%M:%S")
```

As the dataset grows, the "test" window (`train_df[time >= last_date]`) becomes increasingly stale. Training is always cut off at the same date regardless of when the notebook is run.

**Fix:** make `last_date` dynamic, e.g.:
```python
last_date = train_df["time"].max() - pd.to_timedelta(90, unit="D")
```
or accept it as a parameter so it can be set once per training run.
