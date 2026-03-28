# MLOps Batch Signal Generator — Task 0

A minimal MLOps-style batch job that reads OHLCV data, computes a rolling mean on the `close` column, generates a binary trading signal, and outputs structured metrics and logs.

## Local Setup

### Prerequisites

- Python 3.9+
- pip

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run

```bash
python run.py --input data.csv --config config.yaml --output metrics.json --log-file run.log
```

## Docker

### Build

```bash
docker build -t mlops-task .
```

### Run

```bash
docker run --rm mlops-task
```

## Configuration

`config.yaml` controls the pipeline:

| Field    | Description                        |
|----------|------------------------------------|
| `seed`   | Random seed for reproducibility    |
| `window` | Rolling mean window size           |
| `version`| Version tag included in metrics    |

## Signal Logic

- Rolling mean is computed on `close` with the configured `window`.
- The first `window - 1` rows produce NaN rolling means and are assigned `signal = 0`.
- For all other rows: `signal = 1` if `close > rolling_mean`, else `signal = 0`.
- `signal_rate` is computed only over rows with valid (non-NaN) rolling means.

## Example Output

`metrics.json`:

```json
{
  "version": "v1",
  "rows_processed": 10000,
  "metric": "signal_rate",
  "value": 0.4991,
  "latency_ms": 38,
  "seed": 42,
  "status": "success"
}
```
