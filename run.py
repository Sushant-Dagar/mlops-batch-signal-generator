"""MLOps-style batch job: rolling mean signal generation on OHLCV data."""

import argparse
import json
import logging
import sys
import time

import numpy as np
import pandas as pd
import yaml

from backtest import BacktestConfig, config_dict, run_backtest, walk_forward
from signals import rolling_mean_signal


def setup_logging(log_file: str) -> logging.Logger:
    logger = logging.getLogger("mlops_task")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    file_handler = logging.FileHandler(log_file, mode="w")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler(sys.stderr)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    return logger


def load_config(config_path: str, logger: logging.Logger) -> dict:
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    if not isinstance(config, dict):
        raise ValueError("Config file must contain a YAML mapping")

    required_fields = {"seed", "window", "version"}
    missing = required_fields - set(config.keys())
    if missing:
        raise ValueError(f"Config missing required fields: {missing}")

    if not isinstance(config["seed"], int):
        raise ValueError("'seed' must be an integer")
    if not isinstance(config["window"], int) or config["window"] < 1:
        raise ValueError("'window' must be a positive integer")
    if not isinstance(config["version"], str):
        raise ValueError("'version' must be a string")

    logger.info(
        "Config loaded — seed=%d, window=%d, version=%s",
        config["seed"], config["window"], config["version"],
    )
    return config


def load_data(input_path: str, logger: logging.Logger) -> pd.DataFrame:
    # Handle CSVs where entire rows are wrapped in quotes
    import io
    with open(input_path, "r") as f:
        cleaned = f.read().replace('"', '')
    df = pd.read_csv(io.StringIO(cleaned))

    if df.empty:
        raise ValueError("Input CSV is empty")

    if "close" not in df.columns:
        raise ValueError(
            f"Missing required column 'close'. Found columns: {list(df.columns)}"
        )

    logger.info("Loaded %d rows from %s", len(df), input_path)
    return df


def compute_signal(df: pd.DataFrame, window: int, logger: logging.Logger) -> pd.Series:
    signal = rolling_mean_signal(df["close"], window)
    valid = df["close"].rolling(window=window).mean().notna()
    logger.info("Computed rolling mean with window=%d", window)
    logger.info(
        "Generated signals — %d valid rows (excluded %d leading NaN rows)",
        valid.sum(), (~valid).sum(),
    )
    return signal


def run_backtests(df: pd.DataFrame, signal: pd.Series, config: dict,
                  logger: logging.Logger) -> dict:
    """Backtest the signal and, if configured, run a walk-forward test."""
    raw = config.get("backtest") or {}
    bt_cfg = BacktestConfig.from_dict(raw)
    result = run_backtest(df["close"], signal, bt_cfg)
    out = {"config": config_dict(bt_cfg), "metrics": result["metrics"]}
    m = result["metrics"]
    logger.info(
        "Backtest — return=%.2f%%, sharpe=%.2f, max_dd=%.2f%%, trades=%d, costs=%.2f%%",
        m["total_return_pct"], m["sharpe_annualized"], m["max_drawdown_pct"],
        m["num_trades"], m["total_cost_pct"],
    )

    wf = raw.get("walk_forward")
    if wf:
        out["walk_forward"] = walk_forward(
            df["close"], list(wf["windows"]), int(wf["folds"]), bt_cfg)
        logger.info(
            "Walk-forward — out-of-sample sharpe=%.2f over %d folds",
            out["walk_forward"]["out_of_sample_sharpe_annualized"], int(wf["folds"]),
        )
    return out


def write_metrics(output_path: str, metrics: dict):
    with open(output_path, "w") as f:
        json.dump(metrics, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="MLOps batch signal generator")
    parser.add_argument("--input", required=True, help="Path to input CSV")
    parser.add_argument("--config", required=True, help="Path to config YAML")
    parser.add_argument("--output", required=True, help="Path to output metrics JSON")
    parser.add_argument("--log-file", required=True, help="Path to log file")
    args = parser.parse_args()

    logger = setup_logging(args.log_file)
    logger.info("=== Job started ===")

    version = "unknown"

    try:
        start_time = time.time()

        config = load_config(args.config, logger)
        version = config["version"]

        np.random.seed(config["seed"])
        logger.info("Random seed set to %d", config["seed"])

        df = load_data(args.input, logger)

        signal = compute_signal(df, config["window"], logger)

        # Only count rows where rolling mean was valid for signal_rate
        valid_mask = df["close"].rolling(window=config["window"]).mean().notna()
        signal_rate = round(float(signal[valid_mask].mean()), 4)
        rows_processed = len(df)

        elapsed_ms = round((time.time() - start_time) * 1000)

        metrics = {
            "version": version,
            "rows_processed": rows_processed,
            "metric": "signal_rate",
            "value": signal_rate,
            "latency_ms": elapsed_ms,
            "seed": config["seed"],
            "status": "success",
        }

        if "backtest" in config:
            metrics["backtest"] = run_backtests(df, signal, config, logger)
            metrics["latency_ms"] = round((time.time() - start_time) * 1000)

        write_metrics(args.output, metrics)
        logger.info(
            "Metrics — rows=%d, signal_rate=%.4f, latency=%dms",
            rows_processed, signal_rate, elapsed_ms,
        )
        logger.info("=== Job completed successfully ===")

        # Print metrics to stdout for Docker
        print(json.dumps(metrics, indent=2))

    except Exception as exc:
        logger.exception("Job failed: %s", exc)
        error_metrics = {
            "version": version,
            "status": "error",
            "error_message": str(exc),
        }
        write_metrics(args.output, error_metrics)
        print(json.dumps(error_metrics, indent=2))
        sys.exit(1)


if __name__ == "__main__":
    main()
