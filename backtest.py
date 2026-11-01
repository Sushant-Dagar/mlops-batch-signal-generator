"""Event-driven backtester for the rolling-mean signal.

Execution model
---------------
* Long/flat strategy. The signal computed on bar t's close is traded at bar t+1,
  so a position never earns the return of the bar that produced its signal
  (no look-ahead).
* Every change in position pays ``fee_bps + slippage_bps`` on the notional traded.
* Risk controls:
    - ``max_position``: fraction of equity allocated when long (0 < x <= 1).
    - ``stop_loss_pct``: a long trade is closed once its price falls this far
      below the entry price; the strategy then waits for the signal to reset
      to 0 before it can re-enter.
    - ``max_drawdown_pct``: kill switch. Once equity falls this far below its
      peak, the position is closed and the strategy stops trading for the rest
      of the run.
"""

from dataclasses import dataclass, asdict
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from signals import rolling_mean_signal

MINUTES_PER_YEAR = 365 * 24 * 60  # crypto trades 24/7


@dataclass
class BacktestConfig:
    fee_bps: float = 1.0
    slippage_bps: float = 1.0
    max_position: float = 1.0
    stop_loss_pct: Optional[float] = None
    max_drawdown_pct: Optional[float] = None
    periods_per_year: int = MINUTES_PER_YEAR

    @classmethod
    def from_dict(cls, d: Optional[dict]) -> "BacktestConfig":
        d = dict(d or {})
        unknown = set(d) - set(cls.__dataclass_fields__) - {"walk_forward"}
        if unknown:
            raise ValueError(f"Unknown backtest config fields: {sorted(unknown)}")
        d.pop("walk_forward", None)
        cfg = cls(**d)
        cfg.validate()
        return cfg

    def validate(self):
        if self.fee_bps < 0 or self.slippage_bps < 0:
            raise ValueError("fee_bps and slippage_bps must be >= 0")
        if not 0 < self.max_position <= 1:
            raise ValueError("max_position must be in (0, 1]")
        for name in ("stop_loss_pct", "max_drawdown_pct"):
            v = getattr(self, name)
            if v is not None and not 0 < v < 1:
                raise ValueError(f"{name} must be in (0, 1) or null")
        if self.periods_per_year <= 0:
            raise ValueError("periods_per_year must be positive")


def run_backtest(close: pd.Series, signal: pd.Series, cfg: BacktestConfig) -> Dict:
    """Simulate the strategy and return metrics plus the per-bar equity curve."""
    px = close.to_numpy(dtype=float)
    sig = signal.to_numpy(dtype=int)
    n = len(px)
    if n < 2:
        raise ValueError("Need at least 2 bars to backtest")

    cost_rate = (cfg.fee_bps + cfg.slippage_bps) / 1e4
    bar_ret = np.zeros(n)
    bar_ret[1:] = px[1:] / px[:-1] - 1.0

    equity = np.ones(n)
    position = np.zeros(n)  # fraction of equity held during bar t
    pos = 0.0
    eq = 1.0
    peak = 1.0
    entry_px = None
    stopped_out = False  # waiting for signal to reset after a stop
    killed = False
    kill_bar = None
    trades: List[dict] = []
    total_cost = 0.0
    turnover = 0.0
    stops = 0

    for t in range(1, n):
        # 1) Earn this bar's return on the position decided at the end of t-1.
        eq *= 1.0 + pos * bar_ret[t]
        position[t] = pos

        # 2) Risk checks on the close of bar t.
        peak = max(peak, eq)
        if (not killed and cfg.max_drawdown_pct is not None
                and eq <= peak * (1 - cfg.max_drawdown_pct)):
            killed = True
            kill_bar = t
        hit_stop = (pos > 0 and cfg.stop_loss_pct is not None
                    and px[t] <= entry_px * (1 - cfg.stop_loss_pct))

        # 3) Decide the position to hold over bar t+1.
        if sig[t] == 0:
            stopped_out = False
        if killed:
            target = 0.0
        elif hit_stop:
            target = 0.0
            stopped_out = True
            stops += 1
        elif stopped_out:
            target = 0.0
        else:
            target = cfg.max_position if sig[t] == 1 else 0.0

        if target != pos:
            traded = abs(target - pos)
            cost = traded * cost_rate * eq
            eq -= cost
            total_cost += cost
            turnover += traded
            if pos == 0 and target > 0:
                entry_px = px[t]
                trades.append({"entry_bar": t, "entry_px": px[t]})
            elif pos > 0 and target == 0:
                trades[-1].update(exit_bar=t, exit_px=px[t],
                                  reason="stop" if hit_stop else ("kill" if killed else "signal"))
                entry_px = None
            pos = target
        equity[t] = eq

    # Close any open trade at the last bar (marked to market, no extra cost).
    if trades and "exit_bar" not in trades[-1]:
        trades[-1].update(exit_bar=n - 1, exit_px=px[-1], reason="end")

    metrics = compute_metrics(equity, position, trades, cfg)
    metrics.update(
        total_cost_pct=round(float(total_cost) * 100, 4),
        turnover=round(float(turnover), 2),
        stop_losses_hit=stops,
        kill_switch_triggered=bool(killed),
        kill_switch_bar=kill_bar,
        buy_and_hold_return_pct=round(float(px[-1] / px[0] - 1) * 100, 4),
    )
    return {"metrics": metrics, "equity": equity, "position": position, "trades": trades}


def compute_metrics(equity: np.ndarray, position: np.ndarray, trades: List[dict],
                    cfg: BacktestConfig) -> Dict:
    rets = np.diff(equity) / equity[:-1]
    std = rets.std(ddof=1) if len(rets) > 1 else 0.0
    sharpe = float(rets.mean() / std * np.sqrt(cfg.periods_per_year)) if std > 0 else 0.0
    running_peak = np.maximum.accumulate(equity)
    max_dd = float(((equity - running_peak) / running_peak).min())
    trade_rets = [t["exit_px"] / t["entry_px"] - 1 for t in trades]
    wins = sum(r > 0 for r in trade_rets)
    return {
        "total_return_pct": round(float(equity[-1] - 1) * 100, 4),
        "sharpe_annualized": round(sharpe, 3),
        "max_drawdown_pct": round(max_dd * 100, 4),
        "num_trades": len(trades),
        "hit_rate": round(float(wins) / len(trades), 4) if trades else 0.0,
        "exposure": round(float((position > 0).mean()), 4),
    }


def walk_forward(close: pd.Series, windows: List[int], n_folds: int,
                 cfg: BacktestConfig) -> Dict:
    """Anchored walk-forward: pick the window with the best in-sample Sharpe on all
    data before each test fold, then score it on that unseen fold."""
    if n_folds < 1:
        raise ValueError("n_folds must be >= 1")
    if not windows:
        raise ValueError("walk_forward.windows must not be empty")
    n = len(close)
    fold_size = n // (n_folds + 1)
    if fold_size < max(windows) + 2:
        raise ValueError("Not enough data for the requested folds and windows")

    folds = []
    oos_equity = [1.0]
    for k in range(1, n_folds + 1):
        train = close.iloc[: k * fold_size].reset_index(drop=True)
        test_end = n if k == n_folds else (k + 1) * fold_size
        scores = {}
        for w in windows:
            r = run_backtest(train, rolling_mean_signal(train, w), cfg)
            scores[w] = r["metrics"]["sharpe_annualized"]
        best = max(scores, key=scores.get)
        # Compute the signal with history so the test fold starts warm, then slice.
        hist = close.iloc[: test_end].reset_index(drop=True)
        sig = rolling_mean_signal(hist, best).iloc[k * fold_size:].reset_index(drop=True)
        test = hist.iloc[k * fold_size:].reset_index(drop=True)
        res = run_backtest(test, sig, cfg)
        oos_equity.extend((res["equity"][1:] * oos_equity[-1]).tolist())
        folds.append({
            "fold": k,
            "train_bars": len(train),
            "test_bars": len(test),
            "chosen_window": best,
            "in_sample_sharpe": scores[best],
            "out_of_sample": res["metrics"],
        })

    eq = np.array(oos_equity)
    oos = compute_metrics(eq, np.ones(len(eq)), [], cfg)
    return {
        "folds": folds,
        "out_of_sample_total_return_pct": oos["total_return_pct"],
        "out_of_sample_sharpe_annualized": oos["sharpe_annualized"],
        "out_of_sample_max_drawdown_pct": oos["max_drawdown_pct"],
    }


def config_dict(cfg: BacktestConfig) -> dict:
    return asdict(cfg)
