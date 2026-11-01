# MLOps Batch Signal Generator

An MLOps-style batch job that reads OHLCV data, computes a rolling mean on the `close` column, generates a binary trading signal, backtests it with transaction costs and risk controls, and outputs structured metrics and logs.

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
| `backtest` | Optional. Backtest and walk-forward settings (below). Remove it to run the signal job only. |

### Backtest settings

| Field | Description |
|-------|-------------|
| `fee_bps` | Exchange fee per side, in basis points |
| `slippage_bps` | Assumed slippage per side, in basis points |
| `max_position` | Fraction of equity held when long, in (0, 1] |
| `stop_loss_pct` | Close a trade this far below its entry price; wait for the signal to reset before re-entering |
| `max_drawdown_pct` | Kill switch: flatten and stop trading once equity falls this far below its peak |
| `walk_forward.windows` | Rolling-mean windows to choose from |
| `walk_forward.folds` | Number of out-of-sample test folds |

## Signal Logic

- Rolling mean is computed on `close` with the configured `window`.
- The first `window - 1` rows produce NaN rolling means and are assigned `signal = 0`.
- For all other rows: `signal = 1` if `close > rolling_mean`, else `signal = 0`.
- `signal_rate` is computed only over rows with valid (non-NaN) rolling means.

## Backtest

`backtest.py` simulates a long/flat strategy bar by bar:

- **No look-ahead.** A signal computed on bar *t*'s close is traded at bar *t+1*, so a position never earns the return of the bar that produced it.
- **Costs.** Every change in position pays `fee_bps + slippage_bps` on the notional traded.
- **Risk controls.** Position cap, per-trade stop-loss, and a max-drawdown kill switch.
- **Metrics.** Total return, annualized Sharpe ratio (525,600 one-minute bars per year, since crypto trades 24/7), max drawdown, number of trades, hit rate, exposure, turnover, total costs, and a buy-and-hold benchmark.
- **Walk-forward test.** Anchored: before each test fold, the window with the best in-sample Sharpe is chosen using only earlier data, then scored on the unseen fold.

### Results on `data.csv` (10,000 one-minute BTC bars)

| Run | Return | Sharpe | Max drawdown | Trades | Costs |
|-----|--------|--------|--------------|--------|-------|
| Buy and hold | -2.10% | — | — | — | — |
| Window 5, no costs | -13.11% | — | — | 1,390 | 0 |
| Window 5, 2 bps per side, no risk controls | -50.16% | -60.8 | -50.27% | 1,390 | 41.21% |
| Window 5, 2 bps per side, stop-loss + 10% kill switch | -9.88% | -19.2 | -10.08% | 309 | 11.74% |
| Walk-forward, 4 folds, out of sample | -11.41% | -14.6 | -11.85% | — | — |

What this shows:

- The signal has no edge on this data: it loses even before costs.
- At one-minute frequency, trading costs dominate. Costs reach 41% of starting capital, more than three times the pre-cost loss.
- The kill switch capped the drawdown near its 10% limit, against a 50% drawdown without it.
- In-sample selection overfits. The walk-forward chose window 240 every time; its in-sample Sharpe reached 25.0 in fold 1 but was negative in every out-of-sample fold.

## Tests

```bash
pip install -r requirements.txt -r requirements-dev.txt
python -m pytest -q
```

The suite (19 tests) checks the signal definition, no look-ahead, cost accounting, position sizing, stop-loss and kill-switch behaviour, metric calculations, walk-forward data separation, and the end-to-end job. It runs in CI on every push.

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
  "status": "success",
  "backtest": {
    "config": { "fee_bps": 1.0, "slippage_bps": 1.0, "...": "..." },
    "metrics": { "total_return_pct": -9.8788, "sharpe_annualized": -19.163, "max_drawdown_pct": -10.0781, "...": "..." },
    "walk_forward": { "folds": ["..."], "out_of_sample_sharpe_annualized": -14.568 }
  }
}
```
