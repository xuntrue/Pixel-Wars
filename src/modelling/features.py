import numpy as np
import pandas as pd

from src.api.loader import Dataset

# Every feature is "army A minus army B" (or a log ratio), so swapping the armies flips its sign.
# Only information known BEFORE the battle is used: no survivors, no damage dealt.

FEATURE_SETS = {
    "hp_only": ["log_hp_ratio"],
    "size_only": ["log_size_ratio"],
    "core": ["log_hp_ratio", "log_atk_ratio", "log_size_ratio"],

    "core+rates": ["log_hp_ratio", "log_atk_ratio", "log_size_ratio",
                   "log_atk_rate_ratio", "attack_speed_diff", "move_speed_diff"],

    "all": ["log_hp_ratio", "log_atk_ratio", "log_size_ratio",
            "log_atk_rate_ratio", "attack_speed_diff", "move_speed_diff",
            "artifact_diff"],
}
ALL_FEATURES = FEATURE_SETS["all"]      # aggregate features only: never NaN-free by accident, used for dropna
ID_COLS = ["battle_id", "y", "matchup"]


def matchup_groups(ds: Dataset) -> pd.Series:
    """
    battle_id -> text key identifying the matchup (identical armies and artifacts, either side)
    Used so repeats of the same matchup (and its mirror image) never end up in both the training and the test fold
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


# ---------------------------------------------------------------- army composition

def _side_matrix(rows, id_col, side, battle_ids, ids, value_col=None):
    """(n_battles, n_ids) matrix for one side: the count column if given, else 1 where present."""
    sub = rows[rows["side"] == side]
    if sub.empty:
        return np.zeros((len(battle_ids), len(ids)))
    if value_col is None:
        sub = sub.assign(_v=1.0)
        value_col = "_v"
    m = (sub.pivot_table(index="battle_id", columns=id_col, values=value_col,
                         aggfunc="sum", fill_value=0)
            .reindex(index=battle_ids, columns=ids, fill_value=0))
    return m.to_numpy(dtype=float)

def army_vectors(ds: Dataset, battle_ids, unit_ids=None):
    """Return (XA, XB, unit_ids): deployed counts per unit type, rows in battle_ids order."""
    battle_ids = list(battle_ids)
    if unit_ids is None:
        unit_ids = sorted(ds.units["unit_id"])
    unit_ids = list(unit_ids)
    ur = ds.unit_rows[ds.unit_rows["battle_id"].isin(battle_ids)]
    return (_side_matrix(ur, "unit_id", "A", battle_ids, unit_ids, "count"),
            _side_matrix(ur, "unit_id", "B", battle_ids, unit_ids, "count"),
            unit_ids)

def artifact_vectors(ds: Dataset, battle_ids, artifact_ids=None):
    """Return (HA, HB, artifact_ids): 1.0 where the side carried the artifact, else 0.0."""
    battle_ids = list(battle_ids)
    if artifact_ids is None:
        artifact_ids = sorted(ds.artifacts["artifact_id"])
    artifact_ids = list(artifact_ids)
    ar = ds.artifact_rows[ds.artifact_rows["battle_id"].isin(battle_ids)]
    return (np.minimum(_side_matrix(ar, "artifact_id", "A", battle_ids, artifact_ids), 1.0),
            np.minimum(_side_matrix(ar, "artifact_id", "B", battle_ids, artifact_ids), 1.0),
            artifact_ids)

def _frame(values, prefix, ids):
    return pd.DataFrame(values, columns=[f"{prefix}_{i}" for i in ids])

def _antisym_pairs(SA, SB, prefix, ids):
    """For every i < j: SA_i*SB_j - SA_j*SB_i. Swapping the armies flips the sign."""
    i, j = np.triu_indices(SA.shape[1], k=1)
    values = SA[:, i] * SB[:, j] - SA[:, j] * SB[:, i]
    return pd.DataFrame(values, columns=[f"{prefix}_{ids[a]}_{ids[b]}" for a, b in zip(i, j)])

def _artifact_unit_cross(HA, HB, SA, SB, prefix, art_ids, unit_ids, vs_enemy):
    """
    vs_enemy=True : artifact k on A against enemy unit u on B, minus the mirror.
    vs_enemy=False: artifact k alongside its own army's unit u, A minus B.
    """
    if vs_enemy:
        v = HA[:, :, None] * SB[:, None, :] - HB[:, :, None] * SA[:, None, :]
    else:
        v = HA[:, :, None] * SA[:, None, :] - HB[:, :, None] * SB[:, None, :]
    names = [f"{prefix}_{k}_{u}" for k in art_ids for u in unit_ids]
    return pd.DataFrame(v.reshape(len(HA), -1), columns=names)

def composition_features_from_arrays(XA, XB, HA, HB, unit_ids, art_ids) -> pd.DataFrame:
    """
    THE single place that turns "what is on each side" into model inputs. Training calls it via
    composition_features(); the UI should call it directly with the two armies the user entered.
    XA/XB: unit counts (n, n_units). HA/HB: artifact flags (n, n_artifacts).
    """
    SA = XA / np.maximum(XA.sum(axis=1, keepdims=True), 1.0)     # share of each army, by unit count
    SB = XB / np.maximum(XB.sum(axis=1, keepdims=True), 1.0)
    return pd.concat([
        _frame(np.log1p(XA) - np.log1p(XB), "unit", unit_ids),
        _frame(HA - HB, "art", art_ids),
        _antisym_pairs(SA, SB, "upair", unit_ids),
        _antisym_pairs(HA, HB, "apair", art_ids),
        _artifact_unit_cross(HA, HB, SA, SB, "avu", art_ids, unit_ids, vs_enemy=True),
        _artifact_unit_cross(HA, HB, SA, SB, "awu", art_ids, unit_ids, vs_enemy=False),
        _frame(SA, "A_unit", unit_ids), _frame(SB, "B_unit", unit_ids),   # raw sides, for trees
        _frame(HA, "A_art", art_ids), _frame(HB, "B_art", art_ids),
    ], axis=1)

def composition_features(ds: Dataset, battle_ids, unit_ids=None, artifact_ids=None) -> pd.DataFrame:
    """Read the armies of recorded battles and build the composition features for them."""
    battle_ids = list(battle_ids)
    XA, XB, unit_ids = army_vectors(ds, battle_ids, unit_ids)
    HA, HB, art_ids = artifact_vectors(ds, battle_ids, artifact_ids)
    return composition_features_from_arrays(XA, XB, HA, HB, unit_ids, art_ids)

# ---------------------------------------------------------------- feature table
def build_features(ds: Dataset, include_draws: bool = True, composition: bool = True) -> pd.DataFrame:
    """One row per decisive battle: y (1 = A won), matchup key, aggregate features, composition features."""
    b = ds.battles
    d = b[b["winner"] != "draw"] if not include_draws else b      # filter out draws if requested

    out = pd.DataFrame({
        "battle_id": d["battle_id"],
        "y": (d["winner"] == "A").astype(int),
        # log1p keeps armies with a 0 total (e.g. zero attack) finite.
        "log_hp_ratio": np.log1p(d["a_total_hp"]) - np.log1p(d["b_total_hp"]),
        "log_atk_ratio": np.log1p(d["a_total_atk"]) - np.log1p(d["b_total_atk"]),
        "log_size_ratio": np.log(d["a_units"]) - np.log(d["b_units"]),
        "log_atk_rate_ratio": np.log1p(d["a_atk_x_rate"]) - np.log1p(d["b_atk_x_rate"]),  # attack x attack-speed rank
        "attack_speed_diff": d["a_avg_attack_speed"] - d["b_avg_attack_speed"],
        "move_speed_diff": d["a_avg_speed"] - d["b_avg_speed"],
        "artifact_diff": d["a_artifacts"] - d["b_artifacts"],
    })
    out["matchup"] = out["battle_id"].map(matchup_groups(ds))
    out = out.reset_index(drop=True)
    if composition:
        out = pd.concat([out, composition_features(ds, out["battle_id"])], axis=1)
    return out

def _cols(df, prefix, min_support):
    """ Columns starting with prefix that are non-zero in at least min_support battles """
    return [c for c in df.columns if c.startswith(prefix) and (df[c] != 0).sum() >= min_support]

def feature_sets(df: pd.DataFrame, min_support: int = 20) -> dict[str, list[str]]:
    """
    The aggregate sets plus the composition sets, resolved against the columns that exist in df.
    min_support drops columns that are non-zero in fewer than that many battles: a pattern seen
    in 3 battles is noise. Save the resulting column list with the model (save_model does).
    """
    base = FEATURE_SETS["core+rates"]
    units = _cols(df, "unit_", min_support)
    arts = _cols(df, "art_", min_support)
    upairs = _cols(df, "upair_", min_support)
    apairs = _cols(df, "apair_", min_support)
    avu = _cols(df, "avu_", min_support)
    awu = _cols(df, "awu_", min_support)
    raw = _cols(df, "A_", min_support) + _cols(df, "B_", min_support)

    sets = dict(FEATURE_SETS)
    sets.update({
        "units": FEATURE_SETS["all"] + units,                      # which unit types are present
        "arts": base + arts,                                       # which artifacts, individually
        "units+arts": base + units + arts,                         # artifact_diff dropped: it is their sum
        "unit_matchups": base + units + arts + upairs,             # + unit-vs-unit counters
        "full_matchups": base + units + arts + upairs + apairs + avu + awu,
        "trees_raw": FEATURE_SETS["all"] + raw,                    # tree models only; augment with mirror()
    })
    return sets

# Exception: the raw per-side columns (A_*, B_*) made for tree models
def mirror(df: pd.DataFrame) -> pd.DataFrame:
    """
    The same battles with armies A and B swapped: A_*/B_* columns exchanged, every other feature
    negated, y flipped. Use for side-swap augmentation of models that are not symmetric by
    construction (trees, SVM). Augment training folds only.
    """
    out = df.copy()
    for a in [c for c in df.columns if c.startswith("A_")]:
        b = "B_" + a[2:]
        out[a], out[b] = df[b], df[a]
    anti = [c for c in df.columns if c not in ID_COLS and not c.startswith(("A_", "B_"))]
    out[anti] = -df[anti]
    out["y"] = 1 - df["y"]
    return out

def describe_feature(col: str, ds: Dataset) -> str:
    """Readable name for a feature column, for coefficient tables and reports."""
    u = dict(zip(ds.units["id"], ds.units["name"]))
    a = dict(zip(ds.artifacts["id"], ds.artifacts["name"]))
    p = col.split("_")
    try:
        if p[0] == "unit":   return f"{u[int(p[1])]}: A's amount vs B's"
        if p[0] == "art":    return f"{a[int(p[1])]}: A has it vs B has it"
        if p[0] == "upair":  return f"{u[int(p[1])]} (A) vs {u[int(p[2])]} (B), minus the mirror"
        if p[0] == "apair":  return f"{a[int(p[1])]} (A) vs {a[int(p[2])]} (B), minus the mirror"
        if p[0] == "avu":    return f"{a[int(p[1])]} on A vs enemy {u[int(p[2])]}, minus the mirror"
        if p[0] == "awu":    return f"{a[int(p[1])]} with its own {u[int(p[2])]}, A minus B"
        if p[0] in ("A", "B"):
            kind = "unit" if p[1] == "unit" else "artifact"
            return f"side {p[0]}: {(u if kind == 'unit' else a)[int(p[2])]} ({kind})"
    except (KeyError, ValueError, IndexError):
        pass
    return col