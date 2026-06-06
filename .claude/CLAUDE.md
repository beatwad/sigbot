# Sigbot — Project Rules

## What This Project Is

Sigbot is a crypto trading signal bot that:
- Fetches OHLCV candle data from Binance, Bybit, OKEx, and MEXC via exchange APIs
- Computes technical indicators and runs ML models (LightGBM) to detect buy/sell signals
- Sends signal notifications via Telegram
- Runs on a configurable cycle (default: hourly candles, 24-hour window)

## Layout

```
api/            — Exchange API wrappers (one file per exchange/market type)
bot/            — SigBot orchestration (main_cycle logic)
config/         — ConfigFactory + JSON config files
data/           — Candle data loading and caching
indicators/     — Technical indicator computation
ml/             — LightGBM training, inference, stat tests
signal_stat/    — Signal statistics accumulation
signals/        — Signal detection logic
telegram_api/   — Telegram bot wrapper
tests/          — Pytest test suite (fixtures are .pkl files in tests/)
log/            — Loguru log output
```

## Package Manager

Use **uv** for all dependency management.

```bash
uv sync                  # install deps from pyproject.toml
uv add <package>         # add a runtime dependency
uv add --dev <package>   # add a dev dependency
uv run pytest            # run tests inside the venv
```

Never use `pip install` directly; always use `uv`.

## Running the Bot

```bash
uv sync --extra dev          # first-time setup
uv run python main.py        # run the bot (reads config from .env / CONFIG_PATH)
```

The bot loops indefinitely (`while True` in `main.py`). Stop it with `Ctrl+C` — it catches `KeyboardInterrupt` cleanly, terminates the Telegram subprocess, and removes temp image files.

For a debug run against a local config:

```bash
CONFIG_PATH=config/config_debug.json uv run python main.py
```

## Running Tests

```bash
uv run pytest
```

Test fixtures are serialized pandas DataFrames/numpy arrays stored as `.pkl` / `.npy` files in `tests/`. Do not delete or regenerate them without understanding what they contain.

## Code Style

- Line length: **100** (enforced by black + isort)
- Formatter: `black`
- Import sorter: `isort` (profile=black)
- Python target: **3.12**

Run formatters:
```bash
uv run black .
uv run isort .
```

## Config

All runtime behaviour is driven by JSON config files in `config/`. The active config is selected via the `CONFIG_PATH` environment variable (or `.env`). Use `config_test.json` for tests and `config_debug.json` for local runs.

## ML Folder (`ml/`)

### Notebooks

| File | Purpose |
|---|---|
| `prepare_data.ipynb` | Cleans raw signal stat data; filters bad tickers, aligns timestamps against ticker dataframes |
| `select_features.ipynb` | Feature selection — produces `ml/model/features/features.json` |
| `train_model.ipynb` | Trains the LightGBM model — produces `ml/model/lgbm.pkl` |
| `optimize_model.ipynb` | Optuna hyperparameter search — results in `ml/model/optuna/` |
| `test_model.ipynb` | Offline evaluation / backtest of the trained model |
| `inference.py` | In-process inference used by the bot during its main cycle |

### Utils (`ml/utils/`)

| File | Purpose |
|---|---|
| `data_utils.py` | DataFrame loading, merging, splitting helpers |
| `feature_selection_utils.py` | Wrappers for feature importance / selection routines |
| `model_train_utils.py` | LightGBM training helpers |
| `model_test_utils.py` | Evaluation metrics and test helpers |
| `optimization_utils.py` | Optuna objective and study helpers |
| `boosting_uncertainty.py` | Uncertainty estimation for boosting predictions |
| `reparation_utils.py` | Utilities for fixing/patching data issues |
| `stat_tests.py` | Statistical tests (e.g. profitable-hours analysis) |
| `ticker_loader.py` | Loads ticker lists (e.g. from Bybit) for data preparation |

### Model artifacts (`ml/model/`)

| Path | Description |
|---|---|
| `lgbm.pkl` | Trained LightGBM classifier (buy/sell signal quality) |
| `bybit_tickers.json` | List of ~576 Bybit ticker symbols used for training and inference |
| `features/features.json` | Selected feature names, keyed by feature-set index (e.g. `"0"`, `"4"`, ...) |
| `features/feature_importance.csv` | Per-feature importance scores from the trained model |
| `optuna/optuna_lgbm.csv` | Optuna hyperparameter trial results |
| `optuna/optuna_lgbm_info.csv` | Optuna study metadata |

### Signal stat dataframes (`ml/data/signal_stat/`)

These are collected during live bot runs and used as training input.

**`buy_stat_1h.pkl` / `sell_stat_1h.pkl`** — Raw signal statistics (~86K rows, 78 columns each):

| Column | Type | Description |
|---|---|---|
| `time` | datetime64 | Candle timestamp when signal fired |
| `ticker` | str | Trading pair symbol (e.g. `BTCUSDT`) |
| `timeframe` | str | Always `"1h"` in these files |
| `pattern` | str | Signal pattern name (e.g. `STOCH_RSI_Volume24`) |
| `signal_price` | float | Raw price at signal time |
| `signal_smooth_price` | float | Smoothed price at signal time |
| `pct_price_diff_N` | float | % price change N candles after signal (N = 1..24) |
| `mfe_N` | float | Max favorable excursion over N candles |
| `mae_N` | float | Max adverse excursion over N candles |

**`train_buy_272.pkl` / `train_sell_272.pkl`** — Processed training datasets (~72K / ~60K rows, 1566 columns). Built from the raw stat files by joining OHLCV, all technical indicators, and BTC dominance context. Include lookback columns `{feature}_prev_N` for N up to 272 candles. Additional columns:

| Column | Type | Description |
|---|---|---|
| `target` | int (0/1) | Training label — 1 if the signal was profitable |
| `ttype` | str | `"buy"` or `"sell"` |
| `close_time` | datetime64 | Timestamp when the trade closed |
| `max_price_deviation` | float | Max price deviation during the trade |
| `min_price_deviation` | float | Min price deviation during the trade |
| `first_price` | float | Entry price |
| `last_price` | float | Exit price |

`ml/data/signal_stat/backup/` contains backup copies of all four files above.

### Ticker dataframes (`ml/data/tickers/`)

One `.pkl` per ticker per timeframe, named `{TICKER}_{timeframe}.pkl` (e.g. `BTCUSDT_1h.pkl`, `BTCUSDT_4h.pkl`). Each file contains raw OHLCV + funding rate:

| Column | Type | Description |
|---|---|---|
| `time` | datetime64 | Candle open timestamp |
| `open/high/low/close` | float | OHLC prices |
| `volume` | float | Trade volume |
| `funding_rate` | float | Perpetual funding rate (NaN for spot) |

Use `1h` files for signal alignment; `4h` files for multi-timeframe features.

### Context data (`ml/data/context/`)

Market-wide context files joined into training features.

| File | Description |
|---|---|
| `btcd.csv` | BTC dominance daily OHLCV (columns: `time`, `btcd_open/high/low/close/volume`). Historical data from 2014. |
| `btcdom.csv` | BTC.D index 4h OHLCV (columns: `time`, `btcdom_open/high/low/close/volume`). Data from 2021. |
| `profitable_hours.csv` | Output of statistical hour-of-day analysis. Columns: `time`, `test_date`, `TI_low_bound`, `RM_percent_above_0_5`, `profitable_buy_hours_RM`, `profitable_buy_hours_TI`, `profitable_buy_hours` (intersection of RM and TI), `profitable_sell_hours_RM`, `profitable_sell_hours_TI`, `profitable_sell_hours`. Used to filter signal firing times. |

## Important Constraints

- Never commit `.env` — it holds real API keys and Telegram tokens.
- Exchange API calls are rate-limited; avoid calling them in unit tests — use the existing `.pkl` fixtures instead.
- The ML models in `ml/` are trained offline (see `train_model.ipynb`). Inference runs in-process during the bot cycle.
