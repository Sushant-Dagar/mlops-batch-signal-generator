"""MLOps-style batch job: rolling mean signal generation on OHLCV data."""

import argparse
import json
import logging
import sys
import time

import numpy as np
import pandas as pd
import yaml


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
    rolling_mean = df["close"].rolling(window=window).mean()
    logger.info("Computed rolling mean with window=%d", window)

    # Rows where rolling_mean is NaN (first window-1 rows) get signal=0
    signal = (df["close"] > rolling_mean).astype(int)
    signal[rolling_mean.isna()] = 0
    logger.info(
        "Generated signals — %d valid rows (excluded %d leading NaN rows)",
        rolling_mean.notna().sum(),
        rolling_mean.isna().sum(),
    )
    return signal


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
