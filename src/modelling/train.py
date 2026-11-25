from dataclasses import dataclass

import joblib
import numpy as np
import pandas as pd
import re

from src.analysis.stats import line, note, show, subtitle, title
from src.analysis.analyse import show_battle
from src.api.database import DATA_DIR
from src.api.loader import load_default_dataset

from .evaluate import calibration_table, oof_predictions, paired_delta, per_battle_log_loss, summarise
from .features import ALL_FEATURES, build_features, describe_feature, feature_sets
from .models import BASELINES, CANDIDATES, CHAMPION, REFERENCE, VERBOSE

MODEL_DIR = DATA_DIR / "models" # / "win_model.joblib"
SAVE_MODEL = True 

N_SPLITS = 5      # Number of K-folds in the outer CV
N_REPEATS = 5     # Number of repeats of the outer CV (with different random splits each time)
MIN_ROWS = 1_000  # Minimum number of usable battles to train a model

@dataclass
class TrainingData:
    ds: Dataset
    data: pd.DataFrame      # one row per usable decisive battle (features, y, matchup, battle_id)
    y: np.ndarray           # 1 = army A won
    groups: np.ndarray      # matchup id, used to keep a matchup inside one CV fold
    is_real: np.ndarray     # True for real multi-unit battles, False for baseline 1v1s


def load_dataset() -> Dataset:
    title("1. Loading dataset")
    ds = load_default_dataset()
    if ds.battles.empty:
        raise ValueError("No battles recorded yet.")
    line(f"Loaded {len(ds.battles)} battles from {DATA_DIR / 'pixel_wars.db'}")

    n_battles = 5
    cols = ["battle_id", "timestamp", "winner"]
    by_time = ds.battles[cols].sort_values("timestamp")
    subtitle("All Battles Records")
    show(by_time.head(n_battles))
    line("   . . .\n")
    show(by_time.tail(n_battles))
    note(f"Showing {n_battles * 2} of {len(ds.battles)} battles.")

    subtitle("Example battle record")
    real_battles = ds.battles[(ds.battles["a_units"] > 1) & (ds.battles["b_units"] > 1)]
    pool = real_battles if not real_battles.empty else ds.battles   # fall back if none yet
    show_battle(ds, pool.sort_values("timestamp")["battle_id"].iloc[-1])
    return ds

def save_model(name: str, final, features: list[str], trained_on: int, **extra) -> None:
    """ Save a fitted model as models/<name>.joblib, with the details needed to use it later """
    if not SAVE_MODEL:
        return
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    path = MODEL_DIR / f"{name}.joblib"

    joblib.dump(
        {
            "name": name,
            "model": final,
            "features": features,
            "trained_on": trained_on,
            **extra
        },
        path
    )

    line(f"Saved to {path}")

def train() -> None:
    ds = load_dataset()

    title("2. Building features")
    line("Features are built from the armies and artifacts present before the battle, so they can be used to predict the outcome.\n")

    line("Building features. . . ")
    feats = build_features(ds, include_draws=False)
    line(f"Features built for {len(feats)} decisive battles.")

    n_show = 5
    subtitle(f"Showing feature values for the first {n_show} battles")
    show(feats[["battle_id", "y", "matchup"] + ALL_FEATURES].head(n_show).reset_index(drop=True))
    note("matchup is shown as unit_id x count, with each army's artifact ids after the '|'.")

    title("3. Cleaning training data")
    n_before = len(feats)
    data = feats.dropna(subset=ALL_FEATURES + ["matchup"]).reset_index(drop=True)
    line(f"Decisive battles (draws excluded):  {n_before} of {len(ds.battles)} ({100 * n_before / len(ds.battles):.2f}%)")
    line(f"Usable (every feature known):       {len(data)}")
    if len(data) < n_before:
        note(f"{n_before - len(data)} battle(s) dropped. Most likely a unit has no attack speed set.")
    if len(data) < MIN_ROWS:
        line("Not enough usable battles to train yet.")
        return

    # aggregate sets + composition sets, resolved against the columns that exist in data
    fs = feature_sets(data)
    subtitle("Feature sets")
    for name, cols in fs.items():
        shown = ", ".join(cols) if len(cols) <= 8 else ", ".join(cols[:3]) + ", ..."
        line(f"{name:<14} ({len(cols):>4} features): {shown}")

    subtitle("Training data summary")
    y = data["y"].to_numpy()
    groups = pd.factorize(data["matchup"])[0]
    line(f"Distinct matchups (CV groups): {len(set(groups))}")
    line(f"Army A win share (y):          {100 * y.mean():.2f}%")

    # TODO: Move "4. Model training"
    bt = ds.battles.set_index("battle_id")
    real = ((bt["a_units"] > 1) & (bt["b_units"] > 1)).reindex(data["battle_id"]).to_numpy()

    # TODO: Move to analyse.py
    subtitle("Real battles vs baseline 1v1")
    q_real, q_base = y[real].mean(), y[~real].mean()
    h = lambda q: -(q * np.log(q) + (1 - q) * np.log(1 - q))
    line(f"Army A win share in {real.sum()} real battles: {100 * q_real:.2f}%")
    line(f"Predicting own win rate scores: {h(q_real):.4f}")
    line()
    line(f"Army A win share in {(~real).sum()} baseline 1v1: {100 * q_base:.2f}%")
    line(f"Predicting each own win rate scores: {h(q_base):.4f}")
    note("A symmetric model cannot use the A-bias in real battles (it reflects how battles were "
     "recorded, since 1v1s show no side advantage). Its fair yardstick is a coin flip, 0.6931.")

    subtitle("Artifacts line up with winning on real battles")
    rd = data.loc[real]
    g = (rd.assign(artifact_sign=np.sign(rd["artifact_diff"]))
           .groupby("artifact_sign")["y"].agg(battles="size", a_win_rate="mean").reset_index())
    show(g, float_format_level=3)
    line()
    line(f"Army A has more HP in {100 * (rd['log_hp_ratio'] > 0).mean():.1f}% of real battles")

    title("4. Model training")
    # ---- cross-validate every candidate on exactly the same rows and folds ----
    rows, preds_by_name = [], {}
    n_models = len(CANDIDATES)
    n_width = len(str(n_models))
    name_width = max(len(name) for name, _, _ in CANDIDATES)
    set_width = max(len(set_name or "log_hp_ratio only") for _, _, set_name in CANDIDATES)

    line(f"Cross-validating {n_models} candidates")

    for n, (name, factory, set_name) in enumerate(CANDIDATES, start=1):
        label = set_name or "log_hp_ratio only"
        subtitle(f"[{n:>{n_width}}/{n_models}] {name:<{name_width}}  features: {label:<{set_width}}")
   
        features = fs[set_name] if set_name else []
        X = data[features] if features else data[["log_hp_ratio"]]
        preds = oof_predictions(factory() if factory else None, X, y, groups, N_SPLITS, N_REPEATS, verbose=VERBOSE)
        preds_by_name[name] = preds
        rows.append({"model": name,
                     "features": len(features) if name not in ("base rate (A win share)",) else 0,
                     **summarise(y, preds)})

    subtitle(f"Cross-validation ({N_SPLITS} folds x {N_REPEATS} repeats, grouped by matchup)")
    table = pd.DataFrame(rows)
    for col in ("log_loss", "log_loss_sd", "brier"):
        table[col] = table[col].map("{:.4f}".format)
    table["accuracy_pct"] = table["accuracy_pct"].map(lambda v: "-" if pd.isna(v) else f"{v:.2f}%")
    show(table)
    note("log_loss_sd is the spread across CV repeats. accuracy is shown for reference only.")

    subtitle("Log loss by battle type (same out-of-fold predictions)")
    split_rows = []
    for name, preds in preds_by_name.items():
        split_rows.append({
            "model": name,
            "real_multi_unit": f"{per_battle_log_loss(y[real], preds[:, real]).mean():.4f}",
            "baseline_1v1":    f"{per_battle_log_loss(y[~real], preds[:, ~real]).mean():.4f}",
        })
    show(pd.DataFrame(split_rows))
    line(f"Real battles: {real.sum()}   Baseline 1v1: {(~real).sum()}")

    # ---- Are the models actually better than the baseline? ----
    subtitle(f"Log-loss difference vs '{CHAMPION}' (negative = better), 95% interval")
    delta_rows = []
    for name, _, _ in CANDIDATES:
        if name in BASELINES or name == CHAMPION:
            continue
        delta, low, high = paired_delta(y, preds_by_name[name], preds_by_name[CHAMPION], groups)
        delta_rows.append({
            "model": name, "delta": f"{delta:+.4f}",
            "ci_low": f"{low:+.4f}", "ci_high": f"{high:+.4f}",
            "verdict": "better" if high < 0 else "worse" if low > 0 else "no clear difference",
        })
    show(pd.DataFrame(delta_rows))
    note("The interval resamples whole matchups. 'no clear difference' means the "
         "improvement could be luck, so prefer the simpler model.")

    # ---- Best model: calibration, coefficients, save ----
    title("5. Best model selection")
    candidates = [r for r in rows if r["model"] not in BASELINES]
    best_row = min(candidates, key=lambda r: r["log_loss"])
    baseline_loss = min(r["log_loss"] for r in rows if r["model"] in BASELINES)
    if best_row["log_loss"] >= baseline_loss:
        subtitle("Result")
        line("No model beat the baselines. Not saving a model.")
        return

    subtitle("What was our best model selected")
    best_name = best_row["model"]
    _, factory, set_name = next(c for c in CANDIDATES if c[0] == best_name)
    features = fs[set_name]
    line(f"Selected: '{best_name}' ({set_name}, {len(features)} features)")
    line(f"Cross-validated log loss {best_row['log_loss']:.4f}  vs best baseline {baseline_loss:.4f}")

    subtitle(f"Calibration of '{best_name}' (out-of-fold predictions)")
    cal = calibration_table(y, preds_by_name[best_name].mean(axis=0))
    show(cal, float_format_level=3)
    avg_gap = (cal["calibration_gap"] * cal["battles"]).sum() / cal["battles"].sum()
    line(f"Average calibration gap: {100 * avg_gap:.2f}%")
    if avg_gap < 0.03:
        line("Well calibrated: the model's percentages match what actually happened.")
        note(f"With ~{cal['battles'].sum():.2g} battles in {cal.shape[0]} bins, a perfectly calibrated model would still show a gap "
             "of about 1-2% from luck alone, so anything under ~3% is indistinguishable from perfect.")
    elif avg_gap < 0.06:
        line("Reasonably calibrated: the probabilities are usable, with some small distortions.")
        note("Check the bins with the most battles. If the same bins are off in the same direction "
             "in every run, the model is systematically over- or under-confident.")
    elif avg_gap < 0.10:
        line("Noticeably miscalibrated: treat the percentages as rough, not exact.")
        note("Look for a pattern in the table: high bins too low and low bins too high means "
             "overconfident, the opposite means underconfident. Stronger regularisation can help.")
    else:
        line("Poorly calibrated: the model ranks battles sensibly but its percentages are not trustworthy.")
        note("Consider recalibrating (for example isotonic or Platt scaling on the out-of-fold predictions).")
    note("Calibrated = mean_predicted is close to observed_a_win_rate in every row.")

    final = factory().fit(data[features], y)
    subtitle(f"Final model '{best_name}' fitted on all {len(data)} battles")

    steps = getattr(final, "named_steps", {})                  # only logistic regression pipelines have these
    if "clf" in steps and hasattr(steps["clf"], "coef_"):
        clf, scaler = steps["clf"], steps["scale"]
        coefs = pd.DataFrame({
            "feature": features,
            "meaning": [describe_feature(f, ds) for f in features],
            "coef_standardised": clf.coef_[0],
            "coef_raw_units": clf.coef_[0] / scaler.scale_,
        })
        line(f"Chosen regularisation strength C: {clf.C_[0]:.4g}")
        if clf.fit_intercept:
            line(f"Intercept (lean toward army A): {clf.intercept_[0]:+.3f}  "
                 f"(A's odds x{np.exp(clf.intercept_[0]):.2f} when all features are equal)")
        top_n = 15
        coefs = coefs.reindex(coefs["coef_standardised"].abs().sort_values(ascending=False).index)
        show(coefs.head(top_n).reset_index(drop=True))
        if len(coefs) > top_n:
            note(f"Showing the {top_n} largest of {len(coefs)} coefficients, by standardised size.")
        note("coef_standardised compares features with each other. exp(coef_raw_units) is the factor on A's win odds per +1 in that feature (for a log ratio, +1 means A has about 2.7x the quantity).")
    else:
        note("This model has no coefficients to print. Its inputs and outputs are still saved with it.")

    file_name = re.sub(r"[^a-z0-9]+", "_", best_name.lower()).strip("_")   # 'logreg [all]' -> 'logreg_all'
    save_model(file_name, final, features, len(data), cv_log_loss=best_row["log_loss"], feature_set=set_name)
    line()

def train_random_forest() -> None:
    return

def train_grad_boost() -> None:
    return

def train_svm() -> None:
    return
