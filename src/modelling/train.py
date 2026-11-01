import joblib
import pandas as pd

from src.analysis.loader import load_default_dataset
from src.analysis.stats import line, note, show, subtitle, title
from src.api.database import DATA_DIR

from .evaluate import calibration_table, oof_predictions, paired_delta, summarise
from .features import ALL_FEATURES, FEATURE_SETS, build_features
from .models import BASELINES, CANDIDATES, REFERENCE

N_SPLITS = 5
N_REPEATS = 5
SAVE_MODEL = True
MODEL_PATH = DATA_DIR / "models" / "win_model.joblib"
MIN_ROWS = 50


def train() -> None:
    ds = load_default_dataset()
    if ds.battles.empty:
        print("No battles recorded yet.")
        return

    feats = build_features(ds)
    data = feats.dropna(subset=ALL_FEATURES + ["matchup"]).reset_index(drop=True)

    title("Model training: dataset")
    line(f"Decisive battles (draws excluded):  {len(feats)}")
    line(f"Usable (every feature known):       {len(data)}")
    if len(data) < len(feats):
        note(f"{len(feats) - len(data)} battle(s) dropped. Most likely a unit has no attack speed set.")
    if len(data) < MIN_ROWS:
        print("Not enough usable battles to train yet.")
        return

    y = data["y"].to_numpy()
    groups = pd.factorize(data["matchup"])[0]
    line(f"Distinct matchups (CV groups):      {len(set(groups))}")
    line(f"Army A win share:                   {100 * y.mean():.1f}%")

    # ---- cross-validate every candidate on exactly the same rows and folds ----
    rows, preds_by_name = [], {}
    for name, factory, set_name in CANDIDATES:
        features = FEATURE_SETS[set_name] if set_name else []
        X = data[features] if features else data[["log_hp_ratio"]]
        preds = oof_predictions(factory() if factory else None, X, y, groups,
                                N_SPLITS, N_REPEATS)
        preds_by_name[name] = preds
        rows.append({"model": name,
                     "features": len(features) if name not in ("base rate (A win share)",) else 0,
                     **summarise(y, preds)})

    title(f"Cross-validation ({N_SPLITS} folds x {N_REPEATS} repeats, grouped by matchup)")
    table = pd.DataFrame(rows)
    for col in ("log_loss", "log_loss_sd", "brier"):
        table[col] = table[col].map("{:.4f}".format)
    table["accuracy_pct"] = table["accuracy_pct"].map(lambda v: "-" if pd.isna(v) else f"{v:.1f}")
    show(table)
    note("log_loss: 0.6931 is a coin flip, lower is better. log_loss_sd is the spread "
         "across CV repeats. accuracy_pct is shown for reference only.")

    # ---- are the models actually better than the baseline? ----
    subtitle(f"Log-loss difference vs '{REFERENCE}' (negative = better), 95% interval")
    delta_rows = []
    for name, _, _ in CANDIDATES:
        if name in BASELINES:
            continue
        delta, low, high = paired_delta(y, preds_by_name[name], preds_by_name[REFERENCE], groups)
        delta_rows.append({
            "model": name, "delta": f"{delta:+.4f}",
            "ci_low": f"{low:+.4f}", "ci_high": f"{high:+.4f}",
            "verdict": "better" if high < 0 else "worse" if low > 0 else "no clear difference",
        })
    show(pd.DataFrame(delta_rows))
    note("The interval resamples whole matchups. 'no clear difference' means the "
         "improvement could be luck, so prefer the simpler model.")

    # ---- best model: calibration, coefficients, save ----
    candidates = [r for r in rows if r["model"] not in BASELINES]
    best_row = min(candidates, key=lambda r: r["log_loss"])
    baseline_loss = min(r["log_loss"] for r in rows if r["model"] in BASELINES)
    if best_row["log_loss"] >= baseline_loss:
        subtitle("Result")
        line("No model beat the baselines. Not saving a model.")
        return

    best_name = best_row["model"]
    _, factory, set_name = next(c for c in CANDIDATES if c[0] == best_name)
    features = FEATURE_SETS[set_name]

    subtitle(f"Calibration of '{best_name}' (out-of-fold predictions)")
    show(calibration_table(y, preds_by_name[best_name].mean(axis=0)))
    note("Calibrated = mean_predicted is close to observed_a_win_rate in every row.")

    final = factory().fit(data[features], y)
    clf, scaler = final.named_steps["clf"], final.named_steps["scale"]
    coefs = pd.DataFrame({
        "feature": features,
        "coef_standardised": clf.coef_[0],
        "coef_raw_units": clf.coef_[0] / scaler.scale_,
    })
    subtitle(f"Final model '{best_name}' fitted on all {len(data)} battles")
    line(f"Chosen regularisation strength C: {clf.C_[0]:.4g}")
    show(coefs)
    note("coef_standardised compares features with each other. exp(coef_raw_units) is "
         "the factor on A's win odds per +1 in that feature (for a log ratio, +1 means "
         "A has about 2.7x the quantity).")

    if SAVE_MODEL:
        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"model": final, "features": features, "trained_on": len(data)}, MODEL_PATH)
        line(f"Saved to {MODEL_PATH}")
    print()