from pathlib import Path
import pandas as pd


def load_signals(path: str | Path):
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(p)
    return pd.read_csv(p, parse_dates=["timestamp_utc", "target_date"])


def load_settlements(path: str | Path):
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(p)
    return pd.read_csv(p, parse_dates=["target_date"])


def resolve_signal_with_settlement(signal_row: pd.Series, settlement_row: pd.Series):
    # Determine if YES/NO won
    final = settlement_row.get("final_high_f")
    threshold = signal_row.get("threshold_f")
    side = signal_row.get("signal")
    entry_price = signal_row.get("yes_ask") if side == "BUY_YES" else (signal_row.get("no_ask") if side == "BUY_NO" else None)
    if side not in ("BUY_YES", "BUY_NO") or pd.isna(entry_price):
        return None
    # YES wins if final > threshold
    yes_wins = final > threshold
    if side == "BUY_YES":
        pnl = (1 - entry_price) if yes_wins else -entry_price
    else:
        # BUY_NO
        no_wins = not yes_wins
        pnl = (1 - entry_price) if no_wins else -entry_price
    return pnl


def backtest(signals_path: str | Path, settlements_path: str | Path):
    sig = load_signals(signals_path)
    sets = load_settlements(settlements_path)
    results = []
    for _, s in sig.iterrows():
        st = s.get("station")
        td = s.get("target_date")
        matched = sets[(sets["station"] == st) & (sets["target_date"] == pd.to_datetime(td).date())]
        if matched.empty:
            continue
        settlement = matched.iloc[0]
        pnl = resolve_signal_with_settlement(s, settlement)
        if pnl is None:
            continue
        results.append(pnl)
    if not results:
        return {}
    import numpy as np
    arr = np.array(results)
    total = float(arr.sum())
    avg = float(arr.mean())
    win_rate = float((arr > 0).sum() / len(arr))
    # simple max drawdown
    cum = np.cumsum(arr)
    peak = np.maximum.accumulate(cum)
    drawdown = (peak - cum)
    max_dd = float(drawdown.max())
    return {
        "trades": len(arr),
        "total_pnl": total,
        "average_pnl": avg,
        "win_rate": win_rate,
        "max_drawdown": max_dd,
    }
