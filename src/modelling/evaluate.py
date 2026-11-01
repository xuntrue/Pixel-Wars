import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import StratifiedGroupKFold

EPS = 1e-6


def per_battle_log_loss(y, p) -> np.ndarray:
    p = np.clip(p, EPS, 1 - EPS)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def oof_predictions(model, X: pd.DataFrame, y: np.ndarray, groups: np.ndarray,
                    n_splits: int, n_repeats: int, seed: int = 0) -> np.ndarray:
    """
    Out-of-fold P(A wins), shape (n_repeats, n_rows). Every battle is predicted by
    a model that never saw it or any other battle of the same matchup.
    model=None means "always 0.5".
    """
    preds = np.full((n_repeats, len(y)), np.nan)
    if model is None:
        preds[:] = 0.5
        return preds
    for r in range(n_repeats):
        cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed + r)
        for train_idx, test_idx in cv.split(X, y, groups):
            fitted = clone(model).fit(X.iloc[train_idx], y[train_idx])
            preds[r, test_idx] = fitted.predict_proba(X.iloc[test_idx])[:, 1]
    return preds


def summarise(y: np.ndarray, preds: np.ndarray) -> dict:
    """Mean over repeats of each metric (plus the spread of log loss across repeats)."""
    log_losses = [per_battle_log_loss(y, p).mean() for p in preds]
    briers = [np.mean((p - y) ** 2) for p in preds]
    accuracies = []
    for p in preds:
        decided = p != 0.5  # an exact 0.5 makes no prediction
        accuracies.append(np.mean((p[decided] > 0.5) == (y[decided] == 1))
                          if decided.any() else np.nan)
    return {
        "log_loss": float(np.mean(log_losses)),
        "log_loss_sd": float(np.std(log_losses)),
        "brier": float(np.mean(briers)),
        "accuracy_pct": 100 * float(np.nanmean(accuracies)) if not np.isnan(accuracies).all() else np.nan,
    }


def paired_delta(y, preds_a, preds_b, groups, n_boot: int = 2000, seed: int = 0):
    """
    Mean per-battle log-loss difference (a minus b) and a 95% bootstrap interval that
    resamples whole matchups. Negative = model a is better than model b.
    """
    diff = per_battle_log_loss(y, preds_a).mean(axis=0) - per_battle_log_loss(y, preds_b).mean(axis=0)
    by_group = pd.DataFrame({"d": diff, "g": groups}).groupby("g")["d"].agg(["sum", "count"])
    sums, counts = by_group["sum"].to_numpy(), by_group["count"].to_numpy()
    pick = np.random.default_rng(seed).integers(0, len(by_group), size=(n_boot, len(by_group)))
    boot = sums[pick].sum(axis=1) / counts[pick].sum(axis=1)
    low, high = np.percentile(boot, [2.5, 97.5])
    return float(diff.mean()), float(low), float(high)


def calibration_table(y: np.ndarray, p: np.ndarray, bins: int = 10) -> pd.DataFrame:
    """When the model says X%, how often does army A actually win?"""
    cut = pd.cut(p, np.linspace(0, 1, bins + 1), include_lowest=True)
    table = (pd.DataFrame({"p": p, "y": y, "bin": cut})
             .groupby("bin", observed=True)
             .agg(battles=("y", "size"), mean_predicted=("p", "mean"),
                  observed_a_win_rate=("y", "mean"))
             .reset_index())
    table["bin"] = table["bin"].astype(str)
    return table