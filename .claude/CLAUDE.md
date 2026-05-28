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

## Important Constraints

- Never commit `.env` — it holds real API keys and Telegram tokens.
- Exchange API calls are rate-limited; avoid calling them in unit tests — use the existing `.pkl` fixtures instead.
- The ML models in `ml/` are trained offline (see `train_model.ipynb`). Inference runs in-process during the bot cycle.
