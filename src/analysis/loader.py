from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.api.database import ATTACK_SPEEDS


# Per-army aggregates that exist BEFORE the battle (usable as ML features).
PRE_BATTLE_METRICS = ["units", "types", "total_hp", "total_atk", "atk_x_rate", "avg_speed", "avg_attack_speed", "artifacts"]

# Ordinal encoding: Very Slow = 1 ... Very Fast = 5
ATTACK_SPEED_RANK = {name: rank for rank, name in enumerate(ATTACK_SPEEDS, start=1)}

# Per-army aggregates that are OUTCOMES of the battle (never use as features).
RESULT_METRICS = ["survivors", "damage_dealt"]

# Battle ids to leave out
SUSPICIOUS_IDS: set[int] = set([8, 10, 11, 12, 14, 15, 19, 27, 37, 48])

@dataclass
class Dataset:
    units: pd.DataFrame          # catalog: unit_id, name, atk, speed, health
    artifacts: pd.DataFrame      # catalog: artifact_id, name
    battles: pd.DataFrame        # one row per battle, with wide a_* / b_* columns
    sides: pd.DataFrame          # one row per (battle, side)
    unit_rows: pd.DataFrame      # one row per (battle, side, unit group)
    artifact_rows: pd.DataFrame  # one row per (battle, side, artifact)
    excluded: int = 0            # battles dropped because of SUSPICIOUS_IDS

# --- Helper function(s) ---
def _result(winner: pd.Series, side: pd.Series) -> pd.Series:
    """'win' / 'loss' / 'draw' from the point of view of `side`."""
    result = pd.Series("loss", index=winner.index, dtype=object)
    result.loc[winner == "draw"] = "draw"
    result.loc[winner == side] = "win"
    return result

def _sum_if_complete(s: pd.Series) -> float:
    return s.sum() if s.notna().all() else np.nan

def _sum_known(s: pd.Series) -> float:
    return s.sum(min_count=1)  # all-missing stays missing instead of becoming 0

# --- Entry point ---
def load_dataset(conn) -> Dataset:
    units = pd.read_sql_query(
        "SELECT id AS unit_id, name, damage AS atk, speed, attack_speed, health FROM units",
        conn)
    units["attack_speed_rank"] = units["attack_speed"].map(ATTACK_SPEED_RANK) # Ordinal mapping

    artifacts = pd.read_sql_query(
        "SELECT id AS artifact_id, name FROM artifacts", conn)
    battles = pd.read_sql_query(
        "SELECT id AS battle_id, timestamp, winner, carry_over, notes FROM battles", conn)
    bu = pd.read_sql_query("SELECT * FROM battle_units", conn)
    ba = pd.read_sql_query("SELECT * FROM battle_artifacts", conn)

    # Normalise dtypes (empty tables and all-NULL columns come back as 'object').
    battles["timestamp"] = pd.to_datetime(battles["timestamp"], errors="coerce")
    bu = bu.astype({"battle_id": "int64", "unit_id": "int64", "count": "int64"})
    ba = ba.astype({"battle_id": "int64", "artifact_id": "int64"})
    for col in ("remaining", "damage_dealt"):
        bu[col] = pd.to_numeric(bu[col], errors="coerce")

    # Drop battles flagged as suspicious, from every table, before anything is derived.
    excluded = int(battles["battle_id"].isin(SUSPICIOUS_IDS).sum())
    if SUSPICIOUS_IDS:
        battles = battles[~battles["battle_id"].isin(SUSPICIOUS_IDS)].copy()
        bu = bu[~bu["battle_id"].isin(SUSPICIOUS_IDS)].copy()
        ba = ba[~ba["battle_id"].isin(SUSPICIOUS_IDS)].copy()

    winner = battles.set_index("battle_id")["winner"]

    # --- unit groups, enriched ---
    bu = bu.merge(units.rename(columns={"name": "unit"}), on="unit_id", how="left")
    bu["winner"] = bu["battle_id"].map(winner)
    bu["result"] = _result(bu["winner"], bu["side"])
    bu["total_hp"] = bu["count"] * bu["health"]
    bu["total_atk"] = bu["count"] * bu["atk"]
    #bu["atk_x_speed"] = bu["count"] * bu["atk"] * bu["speed"]
    bu["atk_x_rate"] = bu["count"] * bu["atk"] * bu["attack_speed_rank"]
    bu["rank_sum"] = bu["count"] * bu["attack_speed_rank"]
    bu["speed_sum"] = bu["count"] * bu["speed"]
    bu["has_result"] = bu["remaining"].notna() & bu["damage_dealt"].notna()
    bu["deployed_known"] = bu["count"].where(bu["has_result"], 0)

    complete = bu.groupby("battle_id")["has_result"].all()
    battles["complete"] = battles["battle_id"].map(complete).eq(True)

    # --- artifact records, enriched ---
    ba = ba.merge(artifacts.rename(columns={"name": "artifact"}),
                  on="artifact_id", how="left")
    ba["winner"] = ba["battle_id"].map(winner)
    ba["result"] = _result(ba["winner"], ba["side"])

    # --- one row per (battle, side) ---
    sides = bu.groupby(["battle_id", "side"]).agg(
        units=("count", "sum"),
        types=("unit_id", "nunique"),
        total_hp=("total_hp", "sum"),
        total_atk=("total_atk", "sum"),
        atk_x_rate=("atk_x_rate", _sum_if_complete),
        rank_sum=("rank_sum", _sum_if_complete),
        speed_sum=("speed_sum", "sum"),
        survivors=("remaining", _sum_known),
        damage_dealt=("damage_dealt", _sum_known),
    ).reset_index()
    sides["avg_speed"] = sides["speed_sum"] / sides["units"]
    sides["avg_attack_speed"] = sides["rank_sum"] / sides["units"]
    sides = sides.drop(columns=["speed_sum", "rank_sum"])

    incomplete = ~sides["battle_id"].map(complete).astype(bool)
    sides.loc[incomplete, ["survivors", "damage_dealt"]] = np.nan

    if ba.empty:
        sides["artifacts"] = 0
    else:
        counts = ba.groupby(["battle_id", "side"]).size().rename("artifacts").reset_index()
        sides = sides.merge(counts, on=["battle_id", "side"], how="left")
        sides["artifacts"] = sides["artifacts"].fillna(0).astype(int)

    sides["winner"] = sides["battle_id"].map(winner)
    sides["result"] = _result(sides["winner"], sides["side"])
    sides["survivor_pct"] = 100 * sides["survivors"] / sides["units"]

    # --- one row per battle, with a_* / b_* columns side by side ---
    wide = sides.pivot(index="battle_id", columns="side",
                       values=PRE_BATTLE_METRICS + RESULT_METRICS)
    wide.columns = [f"{side.lower()}_{metric}" for metric, side in wide.columns]
    battles = battles.merge(wide, left_on="battle_id", right_index=True, how="left")
    battles["a_win"] = battles["winner"].map({"A": 1.0, "B": 0.0})  # draw -> NaN
    for metric in PRE_BATTLE_METRICS:
        battles[f"d_{metric}"] = battles[f"a_{metric}"] - battles[f"b_{metric}"]
    battles = battles.sort_values(["timestamp", "battle_id"]).reset_index(drop=True)

    return Dataset(units, artifacts, battles, sides, bu, ba, excluded)