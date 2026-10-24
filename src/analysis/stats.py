import math
from textwrap import indent

import pandas as pd

WIDTH = 78

def title(text: str) -> None:
    print("\n" + "=" * WIDTH)
    print(text.upper())
    print("=" * WIDTH)

def subtitle(text: str) -> None:
    print(f"\n--- {text} ---")

def line(text: str = "") -> None:
    print(f"  {text}" if text else "")

def note(text: str) -> None:
    print(f"  Note: {text}")

def pct(part, whole) -> str:
    return f"{100 * part / whole:.1f}%" if whole else "n/a"

def show(df: pd.DataFrame, index: bool = False) -> None:
    """Print a whole DataFrame (never truncated), indented."""
    if df is None or len(df) == 0:
        print("  (no data)")
        return
    text = df.to_string(index=index, float_format="{:,.2f}".format, na_rep="-")
    print(indent(text, "  "))

def wilson(successes: float, n: float, z: float = 1.96) -> tuple[float, float]:
    """ 95% Wilson score interval for a proportion. Much better than the naive interval when the sample is small, which is exactly our situation."""
    if n == 0:
        return float("nan"), float("nan")
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    margin = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return centre - margin, centre + margin


def add_win_rate(df: pd.DataFrame) -> pd.DataFrame:
    """Add win_pct / ci_low / ci_high (in %) from 'wins' and 'losses' columns.
    Draws are excluded from the rate."""
    df = df.copy()
    decisive = df["wins"] + df["losses"]
    bounds = [wilson(w, n) for w, n in zip(df["wins"], decisive)]
    df["win_pct"] = [100 * w / n if n else float("nan")
                     for w, n in zip(df["wins"], decisive)]
    df["ci_low"] = [100 * b[0] for b in bounds]
    df["ci_high"] = [100 * b[1] for b in bounds]
    return df