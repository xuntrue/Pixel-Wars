import numpy as np
import pandas as pd

from src.analysis.loader import Dataset

# Every feature is "army A minus army B" (or a log ratio), so swapping the two
# armies just flips the sign of the row. Only information known BEFORE the
# battle is used: no survivors, no damage dealt.
FEATURE_SETS = {
    "hp_only": ["log_hp_ratio"],
    "size_only": ["log_size_ratio"],
    "core": ["log_hp_ratio", "log_atk_ratio", "log_size_ratio"],
    "core+rates": ["log_hp_ratio", "log_atk_ratio", "log_size_ratio",
                   "log_rate_ratio", "attack_speed_diff", "move_speed_diff"],
    "all": ["log_hp_ratio", "log_atk_ratio", "log_size_ratio",
            "log_rate_ratio", "attack_speed_diff", "move_speed_diff",
            "artifact_diff"],
}
ALL_FEATURES = FEATURE_SETS["all"]


def matchup_groups(ds: Dataset) -> pd.Series:
    """
    battle_id -> text key identifying the matchup (identical armies and
    artifacts, either side). Used so repeats of the same matchup (and its
    mirror image) never end up in both the training and the test fold.
    """
    ur = ds.unit_rows.sort_values(["battle_id", "side", "unit_id"])
    tokens = ur["unit_id"].astype(str) + "x" + ur["count"].astype(str)
    unit_sig = tokens.groupby([ur["battle_id"], ur["side"]]).agg(",".join)

    sig = unit_sig.to_frame("units")
    if ds.artifact_rows.empty:
        sig["arts"] = ""
    else:
        ar = ds.artifact_rows.sort_values(["battle_id", "side", "artifact_id"])
        art_sig = (ar["artifact_id"].astype(str)
                   .groupby([ar["battle_id"], ar["side"]]).agg(",".join))
        sig["arts"] = art_sig.reindex(sig.index).fillna("")

    sig["sig"] = sig["units"] + "|" + sig["arts"]
    wide = sig["sig"].unstack("side")
    a, b = wide["A"], wide["B"]
    return (a + " vs " + b).where(a <= b, b + " vs " + a).rename("matchup")


def build_features(ds: Dataset) -> pd.DataFrame:
    """One row per decisive battle: y (1 = A won), matchup key, and the features."""
    b = ds.battles
    d = b[b["winner"] != "draw"]

    out = pd.DataFrame({
        "battle_id": d["battle_id"],
        "y": (d["winner"] == "A").astype(int),
        # log1p keeps armies with a 0 total (e.g. zero attack) finite.
        "log_hp_ratio": np.log1p(d["a_total_hp"]) - np.log1p(d["b_total_hp"]),
        "log_atk_ratio": np.log1p(d["a_total_atk"]) - np.log1p(d["b_total_atk"]),
        "log_size_ratio": np.log(d["a_units"]) - np.log(d["b_units"]),
        "log_rate_ratio": np.log1p(d["a_atk_x_rate"]) - np.log1p(d["b_atk_x_rate"]),
        "attack_speed_diff": d["a_avg_attack_speed"] - d["b_avg_attack_speed"],
        "move_speed_diff": d["a_avg_speed"] - d["b_avg_speed"],
        "artifact_diff": d["a_artifacts"] - d["b_artifacts"],
    })
    out["matchup"] = out["battle_id"].map(matchup_groups(ds))
    return out.reset_index(drop=True)