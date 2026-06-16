from pathlib import Path
from src.historical_evaluation import evaluate_signals


def main():
    base = Path(__file__).parent.parent
    res = evaluate_signals(base / "data" / "signals.csv", base / "data" / "market_snapshots.csv", base / "data" / "settlements.csv")
    print(res)


if __name__ == "__main__":
    main()
