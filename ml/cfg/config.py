from dataclasses import dataclass


@dataclass(frozen=True)
class SharedCFG:
    """Constants shared across the ml notebooks. Single source of truth.

    Stage-specific knobs (n_folds, train_time_years, test_time_days, ...) stay
    in each notebook's local CFG because they deliberately differ between stages.
    """

    # TP / SL target ratios used to label the data
    TP: float = 0.05
    SL: float = 0.05
    # equivalent multiplicative form used in prepare_data
    target_high_ratio: float = 1.05  # == 1 + TP
    target_low_ratio: float = 0.95  # == 1 - SL
    # per-trade slippage applied in backtests
    slippage: float = 0.002
    # maximum number of simultaneously opened trades for backtest metric
    max_num_simult_trades: int = 100
    # minimum precision threshold for accepting a model
    min_precision: float = 0.5


SHARED = SharedCFG()
