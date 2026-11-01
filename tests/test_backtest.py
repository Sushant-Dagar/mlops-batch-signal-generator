import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backtest import BacktestConfig, compute_metrics, run_backtest, walk_forward  # noqa: E402
from signals import rolling_mean_signal  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
FREE = BacktestConfig(fee_bps=0, slippage_bps=0)


def s(values):
    return pd.Series(values, dtype=float)


# --- signal -------------------------------------------------------------

def test_signal_matches_definition():
    close = s([1, 2, 3, 2, 1, 5])
    sig = rolling_mean_signal(close, 3)
    assert sig.tolist() == [0, 0, 1, 0, 0, 1]


def test_signal_rejects_bad_window():
    with pytest.raises(ValueError):
        rolling_mean_signal(s([1, 2]), 0)


# --- execution ----------------------------------------------------------

def test_no_lookahead_signal_trades_next_bar():
    close = s([100, 100, 110, 110])
    # Signal fires on the jump bar itself: must NOT earn that jump.
    late = run_backtest(close, pd.Series([0, 0, 1, 0]), FREE)
    assert late["metrics"]["total_return_pct"] == pytest.approx(0.0)
    # Signal fires one bar before the jump: earns +10%.
    early = run_backtest(close, pd.Series([0, 1, 0, 0]), FREE)
    assert early["metrics"]["total_return_pct"] == pytest.approx(10.0)


def test_costs_charged_on_entry_and_exit():
    close = s([100, 100, 100, 100])
    cfg = BacktestConfig(fee_bps=5, slippage_bps=5)  # 10 bps per side
    r = run_backtest(close, pd.Series([0, 1, 0, 0]), cfg)
    expected = (1 - 0.001) * (1 - 0.001) - 1
    assert r["metrics"]["total_return_pct"] == pytest.approx(expected * 100, abs=1e-3)
    assert r["metrics"]["turnover"] == 2.0
    assert r["metrics"]["num_trades"] == 1


def test_max_position_scales_exposure():
    close = s([100, 100, 110])
    cfg = BacktestConfig(fee_bps=0, slippage_bps=0, max_position=0.5)
    r = run_backtest(close, pd.Series([0, 1, 1]), cfg)
    assert r["metrics"]["total_return_pct"] == pytest.approx(5.0)


def test_stop_loss_exits_and_waits_for_reset():
    close = s([100, 100, 98, 97, 96, 96, 100, 110])
    sig = pd.Series([0, 1, 1, 1, 1, 0, 1, 1])
    cfg = BacktestConfig(fee_bps=0, slippage_bps=0, stop_loss_pct=0.01)
    r = run_backtest(close, sig, cfg)
    pos = r["position"]
    assert r["metrics"]["stop_losses_hit"] == 1
    assert r["trades"][0]["reason"] == "stop"
    # Stopped at bar 2's close; flat through bars 3-5 despite signal=1 at 3-4.
    assert pos[3] == pos[4] == pos[5] == 0
    # Signal reset at bar 5, re-entered at bar 6's close, long during bar 7.
    assert pos[7] == 1.0
    assert r["metrics"]["num_trades"] == 2


def test_kill_switch_stops_trading():
    close = s([100, 100, 80, 80, 100, 120])
    cfg = BacktestConfig(fee_bps=0, slippage_bps=0, max_drawdown_pct=0.1)
    r = run_backtest(close, pd.Series([1, 1, 1, 1, 1, 1]), cfg)
    m = r["metrics"]
    assert m["kill_switch_triggered"] is True
    assert m["kill_switch_bar"] == 2
    assert (r["position"][3:] == 0).all()
    assert m["total_return_pct"] == pytest.approx(-20.0)


def test_metrics_on_known_curve():
    equity = np.array([1.0, 1.1, 0.99, 1.2])
    m = compute_metrics(equity, np.ones(4), [], FREE)
    assert m["max_drawdown_pct"] == pytest.approx(-10.0)
    assert m["total_return_pct"] == pytest.approx(20.0)


def test_flat_strategy_has_zero_sharpe():
    r = run_backtest(s([100, 101, 102]), pd.Series([0, 0, 0]), FREE)
    assert r["metrics"]["sharpe_annualized"] == 0.0
    assert r["metrics"]["num_trades"] == 0


@pytest.mark.parametrize("bad", [
    {"max_position": 0}, {"max_position": 1.5}, {"fee_bps": -1},
    {"stop_loss_pct": 1.2}, {"max_drawdown_pct": 0}, {"unknown_field": 1},
])
def test_config_validation(bad):
    with pytest.raises((ValueError, TypeError)):
        BacktestConfig.from_dict(bad)


# --- walk-forward -------------------------------------------------------

def test_walk_forward_uses_only_past_data():
    rng = np.random.default_rng(0)
    close = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.001, 1000))))
    wf = walk_forward(close, [5, 20], 3, BacktestConfig())
    assert len(wf["folds"]) == 3
    for k, fold in enumerate(wf["folds"], start=1):
        assert fold["train_bars"] == k * 250
        assert fold["chosen_window"] in (5, 20)
    assert sum(f["test_bars"] for f in wf["folds"]) == 750


def test_walk_forward_rejects_too_little_data():
    with pytest.raises(ValueError):
        walk_forward(s(range(1, 20)), [50], 3, BacktestConfig())


# --- end to end ---------------------------------------------------------

def _run(tmp_path, config_text):
    cfg = tmp_path / "config.yaml"
    cfg.write_text(config_text)
    out = tmp_path / "metrics.json"
    subprocess.run(
        [sys.executable, str(ROOT / "run.py"), "--input", str(ROOT / "data.csv"),
         "--config", str(cfg), "--output", str(out), "--log-file", str(tmp_path / "run.log")],
        check=True, capture_output=True,
    )
    return json.loads(out.read_text())


def test_job_without_backtest_keeps_original_output(tmp_path):
    m = _run(tmp_path, 'seed: 42\nwindow: 5\nversion: "v1"\n')
    assert m["value"] == 0.4991
    assert "backtest" not in m


def test_job_with_backtest(tmp_path):
    m = _run(tmp_path, 'seed: 42\nwindow: 5\nversion: "v2"\n'
                       'backtest:\n  fee_bps: 1\n  slippage_bps: 1\n'
                       '  walk_forward:\n    windows: [5, 20]\n    folds: 2\n')
    assert m["status"] == "success"
    assert m["backtest"]["metrics"]["num_trades"] > 0
    assert len(m["backtest"]["walk_forward"]["folds"]) == 2
