from itertools import combinations

import pandas as pd

from src.api.database import get_connection, init_db
from .loader import ATTACK_SPEED_RANK, Dataset, load_dataset
from .stats import add_win_rate, line, note, pct, show, subtitle, title, wilson

TOP_N = 5        # length of the "top" lists for combinatorial tables
MIN_SAMPLE = 50  # fewer decisive appearances than this -> flagged "low n"

# (column suffix, label) for "does the stronger army win?"
COMPARE = [
    ("units", "Army size (total units)"),
    ("types", "Distinct unit types"),
    ("total_hp", "Total HP"),
    ("total_atk", "Total attack"),
    ("atk_x_rate", "Attack x attack-speed rank"),
    ("avg_speed", "Average move speed"),
    ("avg_attack_speed", "Average attack speed (1=Very Slow, 5=Very Fast)"),
    ("avg_speed", "Average speed"),
    ("artifacts", "Artifact count"),
]

# --- Helper function(s) ---
def _decisive_results(ds: Dataset) -> pd.DataFrame:
    """Decisive battles with full results, with winner_/loser_ columns."""
    b = ds.battles
    d = b[b["complete"] & (b["winner"] != "draw")].copy()
    is_a = d["winner"] == "A"
    for col in ("units", "survivors", "damage_dealt", "total_hp"):
        d[f"win_{col}"] = d[f"a_{col}"].where(is_a, d[f"b_{col}"])
        d[f"lose_{col}"] = d[f"b_{col}"].where(is_a, d[f"a_{col}"])
    d["win_survivor_pct"] = 100 * d["win_survivors"] / d["win_units"]
    d["lose_survivor_pct"] = 100 * d["lose_survivors"] / d["lose_units"]
    d["win_dmg_vs_enemy_hp"] = d["win_damage_dealt"] / d["lose_total_hp"]
    d["lose_dmg_vs_enemy_hp"] = d["lose_damage_dealt"] / d["win_total_hp"]
    return d

def _heuristic_table(ds: Dataset) -> pd.DataFrame:
    """For each metric: how often does the army with the higher value win?"""
    d = ds.battles[ds.battles["winner"] != "draw"]
    rows = []
    for key, label in COMPARE:
        col = f"d_{key}"
        sub = d[d[col].notna() & (d[col] != 0)]
        if sub.empty:
            continue
        correct = int(((sub[col] > 0) == (sub["a_win"] == 1.0)).sum())
        low, high = wilson(correct, len(sub))
        rows.append({
            "metric": label,
            "compared": len(sub),
            "skipped": len(d) - len(sub),
            "higher_wins": correct,
            "accuracy_pct": 100 * correct / len(sub),
            "ci_low": 100 * low,
            "ci_high": 100 * high,
            "corr_with_A_win": d[col].corr(d["a_win"]),
        })
    return pd.DataFrame(rows)

def unit_matchup_matrix(ds: Dataset) -> None:
    """ Baseline: how often has each unit type faced each other unit type, one on one? """
    subtitle("Unit type matchups: single-unit baseline (battle counts)")
    b = ds.battles

    clean = (
        (b["a_types"] == 1) & (b["b_types"] == 1)             # one unit type per army
        & (b["a_artifacts"] == 0) & (b["b_artifacts"] == 0)   # no artifacts      
        & (b["a_units"] == 1) & (b["b_units"] == 1)           # only one unit deployed per army
    )
    ids = b.loc[clean, "battle_id"]
    line(f"Qualifying battles: {len(ids)} of {len(b)}")

    # Rows and columns: every unit type in the catalog, so unseen matchups show up as 0.
    names = ds.units.sort_values("name").set_index("unit_id")["name"]
    n = len(names)
    if n == 0:
        return
    matrix = pd.DataFrame(0, index=names.index, columns=names.index)

    if len(ids):
        pairs = (ds.unit_rows[ds.unit_rows["battle_id"].isin(ids)]
                 .pivot(index="battle_id", columns="side", values="unit_id")
                 .dropna())
        for a_id, b_id in zip(pairs["A"].astype(int), pairs["B"].astype(int)):
            matrix.loc[a_id, b_id] += 1
            if a_id != b_id:
                matrix.loc[b_id, a_id] += 1  # side-swap: the same battle seen from the other side

    show(matrix.rename(index=names, columns=names), index=True)

    total_cells = n * n
    zero_cells = int((matrix.values == 0).sum())
    seen_cells = total_cells - zero_cells
    line()
    line(f"Matrix cells with 0 battles (matchups not yet seen): {zero_cells} of {total_cells}")
    line(f"Matchups seen: {seen_cells} of {total_cells} cells ({pct(seen_cells, total_cells)})")

    # X vs Y and Y vs X are the same fight, so also count each pair once.
    diagonal_seen = int((matrix.values.diagonal() > 0).sum())
    seen_pairs = (seen_cells + diagonal_seen) // 2
    total_pairs = n * (n + 1) // 2
    line(f"Distinct matchups seen (X vs Y = Y vs X): {seen_pairs} of {total_pairs} "
         f"({pct(seen_pairs, total_pairs)})")

    note("Each cell is the number of qualifying battles between those two unit types. "
         "The matrix is symmetric because every battle is counted from both sides (side-swap augmentation).")
    note("The diagonal is mirror matches (the same unit type on both sides)."
         " A mirror  battle fills its cell once, not twice. The total-cell numbers include the diagonal.")

def _cooccurrence(rows: pd.DataFrame, id_col: str, result_of: dict) -> dict:
    """Pairs of items on the same side -> win/loss/draw tallies."""
    key_for = {"win": "wins", "loss": "losses", "draw": "draws"}
    stats = {}
    for (battle_id, side), group in rows.groupby(["battle_id", "side"]):
        outcome = key_for[result_of[(battle_id, side)]]
        for pair in combinations(sorted(group[id_col]), 2):
            stats.setdefault(pair, {"wins": 0, "losses": 0, "draws": 0})[outcome] += 1
    return stats

def _pair_frame(stats: dict, first_names: pd.Series, second_names: pd.Series,
                first: str, second: str) -> pd.DataFrame:
    """Every pair with win/loss/draw tallies and win rates (not sorted or trimmed)."""
    rows = []
    for (x, y), t in stats.items():
        rows.append({first: first_names.get(x, x), second: second_names.get(y, y),
                     "appearances": t["wins"] + t["losses"] + t["draws"], **t})
    if not rows:
        return pd.DataFrame()
    return add_win_rate(pd.DataFrame(rows))


def _most_common(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    return df.sort_values("appearances", ascending=False).head(TOP_N)

def _show_best_win_rate(df: pd.DataFrame) -> None:
    """Highest win rates among combinations with enough decisive appearances."""
    if not df.empty:
        enough = df[(df["wins"] + df["losses"]) >= MIN_SAMPLE]
        best = enough.sort_values(["win_pct", "ci_low"], ascending=False).head(TOP_N)
    else:
        best = df
    if best.empty:
        line(f"No combination has {MIN_SAMPLE}+ decisive appearances yet.")
    else:
        show(best)

def _unit_artifact_cooccurrence(ds: Dataset, result_of: dict) -> dict:
    """(unit, artifact) pairs that were on the same side -> win/loss/draw tallies."""
    key_for = {"win": "wins", "loss": "losses", "draw": "draws"}
    artifacts_by_side = {key: list(g["artifact_id"])
                         for key, g in ds.artifact_rows.groupby(["battle_id", "side"])}
    stats = {}
    for key, group in ds.unit_rows.groupby(["battle_id", "side"]):
        outcome = key_for[result_of[key]]
        for unit_id in group["unit_id"]:
            for artifact_id in artifacts_by_side.get(key, []):
                tally = stats.setdefault((unit_id, artifact_id),
                                         {"wins": 0, "losses": 0, "draws": 0})
                tally[outcome] += 1
    return stats

# ========== Sections ==========
def overview(ds: Dataset) -> None:
    title("1. Dataset overview")
    b = ds.battles
    line(f"Unit types in catalog:      {len(ds.units)}")
    line(f"Artifacts in catalog:       {len(ds.artifacts)}")
    
    if ds.excluded:
        line(f"Battles recorded:           {len(b) + ds.excluded}")
        line(f"Battles excluded:           {ds.excluded}")
        line(f"Battles in dataset:         {len(b)}")
    else:
        line(f"Battles recorded:           {len(b)}")

    line(f"Unit records (rows):        {len(ds.unit_rows)}")
    line(f"Artifact records (rows):    {len(ds.artifact_rows)}")
    line(f"Battles with full results:  {int(b['complete'].sum())} ({pct(b['complete'].sum(), len(b))})")
    line(f"Battles with notes:         {int(b['notes'].notna().sum())}")

    

def timeline(ds: Dataset) -> None:
    title("2. Timeline")
    b = ds.battles
    first, last = b["timestamp"].min(), b["timestamp"].max()
    line(f"First battle: {first}   Last battle: {last}   Span: {last - first}")
    per_day = b.groupby(b["timestamp"].dt.date).size().rename("battles")
    line(f"Active days: {len(per_day)}   Average per active day: {per_day.mean():.1f}"
         f"   Busiest day: {per_day.max()}")
    note("timestamps are stored in UTC.")
    subtitle("Battles per day (last 30 active days)")
    show(per_day.tail(30).rename_axis("date").reset_index())


def outcomes(ds: Dataset) -> None:
    title("3. Outcomes and side balance")
    b = ds.battles
    counts = b["winner"].value_counts()
    for key, label in (("A", "Army A wins"), ("B", "Army B wins"), ("draw", "Draws")):
        c = int(counts.get(key, 0))
        line(f"{label:<12} {c:>5}   ({pct(c, len(b))})")
    a_wins, b_wins = int(counts.get("A", 0)), int(counts.get("B", 0))
    decisive = a_wins + b_wins
    if decisive:
        low, high = wilson(a_wins, decisive)
        line(f"Army A win rate among decisive battles: {pct(a_wins, decisive)} "
             f"(95% CI {100 * low:.1f}% to {100 * high:.1f}%)")
        note("if A/B assignment is arbitrary, ~50% is expected; a persistent gap may mean you tend to put the same kind of army on one side.")

def catalog_usage(ds: Dataset) -> None:
    title("4. Catalog and usage")
    subtitle("Unit catalog: stat summary")
    show(ds.units[["atk", "speed", "health"]].describe(percentiles=[0.25, 0.5, 0.75]).T, index=True)
    subtitle("Unit types: stats and usage (sorted by total deployed)")
    usage = ds.unit_rows.groupby("unit_id").agg(
        battles=("battle_id", "nunique"),
        total_deployed=("count", "sum"),
        avg_deployed=("count", "mean"),
    ).reset_index()
    table = ds.units.merge(usage, on="unit_id", how="left")
    table[["battles", "total_deployed"]] = table[["battles", "total_deployed"]].fillna(0).astype(int)
    table = table.sort_values("total_deployed", ascending=False)
    show(table.drop(columns=["unit_id", "attack_speed_rank"]))
    unused = table.loc[table["battles"] == 0, "name"].tolist()
    line(f"\nUnit types never used: {len(unused)}"
         + (f"  ({', '.join(unused[:30])}{', ...' if len(unused) > 30 else ''})" if unused else ""))

    subtitle("Attack speed mix across the unit catalog")
    mix = ds.units["attack_speed"].fillna("(not set)").value_counts().rename_axis("attack_speed")
    show(mix.reset_index(name="unit_types"))

    subtitle("Artifacts: usage (sorted by times held)")
    if ds.artifact_rows.empty:
        table = ds.artifacts.assign(battles=0, times_held=0)
    else:
        usage = ds.artifact_rows.groupby("artifact_id").agg(
            battles=("battle_id", "nunique"), times_held=("battle_id", "count")
        ).reset_index()
        table = ds.artifacts.merge(usage, on="artifact_id", how="left")
        table[["battles", "times_held"]] = table[["battles", "times_held"]].fillna(0).astype(int)
    table = table.sort_values("times_held", ascending=False)
    show(table.drop(columns="artifact_id"))
    unused = table.loc[table["times_held"] == 0, "name"].tolist()
    line(f"\nArtifacts never used: {len(unused)}" + (f"  ({', '.join(unused[:30])}{', ...' if len(unused) > 30 else ''})" if unused else ""))

def army_composition(ds: Dataset) -> None:
    title("5. Army composition")
    s = ds.sides
    cols = ["units", "types", "total_hp", "total_atk", "atk_x_rate", "avg_speed", "avg_attack_speed", "artifacts"]

    subtitle("Distribution per army (both sides pooled)")
    show(s[cols].describe(percentiles=[0.25, 0.5, 0.75]).T, index=True)

    subtitle("Average per army, by result")
    by_result = s.groupby("result")[cols].mean().T
    show(by_result[[c for c in ("win", "loss", "draw") if c in by_result.columns]], index=True)

    subtitle("Average per army, by side")
    show(s.groupby("side")[cols].mean().T, index=True)

    subtitle("Armies by number of distinct unit types")
    show(s["types"].value_counts().sort_index().rename_axis("distinct_types")
         .reset_index(name="armies"))
    #note("'atk_x_speed' is attack stat x speed x count, a rough damage-rate proxy. Replace it if you learn how speed really works in the game.")
    note("'atk_x_rate' = attack stat x attack-speed rank (1-5) x count, an ordinal proxy for damage rate, not real attacks per second. 'avg_speed' is movement speed.")

def strength_vs_outcome(ds: Dataset) -> None:
    title("6. Does the 'stronger' army win?")
    d = ds.battles[ds.battles["winner"] != "draw"]
    if d.empty:
        line("No decisive battles yet.")
        return

    a_share = (d["winner"] == "A").mean()
    line(f"Decisive battles: {len(d)}")
    line(f"Baseline 'always predict Army A': {100 * max(a_share, 1 - a_share):.1f}% accurate "
         f"(majority side: {'A' if a_share >= 0.5 else 'B'})")
    subtitle("Accuracy of 'the army with the higher value wins'")
    show(_heuristic_table(ds))
    note("'skipped' = battles where the two armies were tied on that metric. "
         "accuracy_pct is your first benchmark: a trained model should beat the best of these.")


def unit_performance(ds: Dataset) -> None:
    title("7. Unit performance")
    t = ds.unit_rows.groupby("unit_id").agg(
        appearances=("battle_id", "count"),
        total_deployed=("count", "sum"),
        wins=("result", lambda r: int((r == "win").sum())),
        losses=("result", lambda r: int((r == "loss").sum())),
        draws=("result", lambda r: int((r == "draw").sum())),
        remaining=("remaining", "sum"),
        deployed_known=("deployed_known", "sum"),
        damage=("damage_dealt", "sum"),
        atk=("atk", "first"),
    ).reset_index()
    known = t["deployed_known"].where(t["deployed_known"] > 0)
    t["survival_pct"] = 100 * t["remaining"] / known
    t["dmg_per_unit"] = t["damage"] / known
    t["dmg_per_unit_per_atk"] = t["dmg_per_unit"] / t["atk"].where(t["atk"] > 0)
    t = add_win_rate(t.merge(ds.units[["unit_id", "name"]], on="unit_id"))
    t["low_n"] = (t["wins"] + t["losses"]) < MIN_SAMPLE
    t["flag"] = t["low_n"].map({True: "low n", False: ""})
    t = t.sort_values(["low_n", "win_pct"], ascending=[True, False])

    show(t[["name", "appearances", "total_deployed", "wins", "losses", "draws",
            "win_pct", "ci_low", "ci_high", "survival_pct", "dmg_per_unit",
            "dmg_per_unit_per_atk", "flag"]])
    note("win_pct = share of decisive battles won by a side that deployed the unit "
         "(95% Wilson CI). It is confounded: a unit that is always in strong armies "
         "looks strong. Treat it as descriptive, not causal.")
    note("survival_pct = survivors / deployed. dmg_per_unit = damage dealt by the group / "
         "units deployed. dmg_per_unit_per_atk divides that by the unit's attack stat, "
         "roughly 'how many attacks landed per unit'.")
    note(f"'low n' = fewer than {MIN_SAMPLE} decisive appearances; those rows sort last.")

    subtitle("Unit performance grouped by attack speed")
    g = ds.unit_rows.assign(attack_speed=ds.unit_rows["attack_speed"].fillna("(not set)"))
    by_speed = g.groupby("attack_speed").agg(
        appearances=("battle_id", "count"),
        wins=("result", lambda r: int((r == "win").sum())),
        losses=("result", lambda r: int((r == "loss").sum())),
        draws=("result", lambda r: int((r == "draw").sum())),
    ).reset_index()
    by_speed["order"] = by_speed["attack_speed"].map(ATTACK_SPEED_RANK)
    by_speed = add_win_rate(by_speed).sort_values("order").drop(columns="order")
    show(by_speed)

    unit_matchup_matrix(ds)

def artifact_performance(ds: Dataset) -> None:
    title("8. Artifact performance")
    ar = ds.artifact_rows
    if ar.empty:
        line("No artifacts have been used in a recorded battle yet.")
    else:
        t = ar.groupby("artifact_id").agg(
            appearances=("battle_id", "count"),
            wins=("result", lambda r: int((r == "win").sum())),
            losses=("result", lambda r: int((r == "loss").sum())),
            draws=("result", lambda r: int((r == "draw").sum())),
        ).reset_index().merge(ds.artifacts, on="artifact_id")
        t = add_win_rate(t)
        t["lift_pct"] = t["win_pct"] - 50
        t["low_n"] = (t["wins"] + t["losses"]) < MIN_SAMPLE
        t["flag"] = t["low_n"].map({True: "low n", False: ""})
        t = t.sort_values(["low_n", "win_pct"], ascending=[True, False])
        show(t[["name", "appearances", "wins", "losses", "draws",
                "win_pct", "ci_low", "ci_high", "lift_pct", "flag"]])
        note("lift_pct is win_pct minus 50: every decisive battle has exactly one winner, "
             "so the average side wins 50% by construction.")

    subtitle("Win rate by number of artifacts an army holds")
    by_count = ds.sides.groupby("artifacts")["result"].agg(
        armies="count",
        wins=lambda r: int((r == "win").sum()),
        losses=lambda r: int((r == "loss").sum()),
        draws=lambda r: int((r == "draw").sum()),
    ).reset_index()
    show(add_win_rate(by_count))

def combinations_section(ds: Dataset) -> None:
    title("9. Combinations")
    result_of = {(r.battle_id, r.side): r.result for r in ds.sides.itertuples()}
    unit_names = ds.units.set_index("unit_id")["name"]
    artifact_names = ds.artifacts.set_index("artifact_id")["name"]

    unit_pairs = _pair_frame(_cooccurrence(ds.unit_rows, "unit_id", result_of),
                             unit_names, unit_names, "unit_1", "unit_2")
    artifact_pairs = _pair_frame(_cooccurrence(ds.artifact_rows, "artifact_id", result_of),
                                 artifact_names, artifact_names, "artifact_1", "artifact_2")
    unit_artifact = _pair_frame(_unit_artifact_cooccurrence(ds, result_of),
                                unit_names, artifact_names, "unit", "artifact")

    subtitle(f"Most common unit pairs in the same army (top {TOP_N})")
    show(_most_common(unit_pairs))

    subtitle(f"Unit pairs in the same army with the highest win rate "
             f"(top {TOP_N}, min {MIN_SAMPLE} decisive appearances)")
    _show_best_win_rate(unit_pairs)

    subtitle(f"Most common unit + artifact combinations in the same army (top {TOP_N})")
    show(_most_common(unit_artifact))

    subtitle(f"Unit + artifact combinations with the highest win rate "
             f"(top {TOP_N}, min {MIN_SAMPLE} decisive appearances)")
    _show_best_win_rate(unit_artifact)

    subtitle(f"Most common artifact pairs in the same army (top {TOP_N})")
    show(_most_common(artifact_pairs))

    subtitle(f"Most common unit matchups across the two armies (top {TOP_N})")
    by_side = {key: sorted(g["unit_id"])
               for key, g in ds.unit_rows.groupby(["battle_id", "side"])}
    key_for = {"win": "wins", "loss": "losses", "draw": "draws"}
    matchups = {}
    for battle_id in ds.battles["battle_id"]:
        for x in by_side.get((battle_id, "A"), []):
            for y in by_side.get((battle_id, "B"), []):
                if x == y:
                    continue
                pair, side = ((x, y), "A") if x < y else ((y, x), "B")
                outcome = key_for[result_of[(battle_id, side)]]
                matchups.setdefault(pair, {"wins": 0, "losses": 0, "draws": 0})[outcome] += 1
    show(_most_common(_pair_frame(matchups, unit_names, unit_names, "unit_1", "vs_unit_2")))

    #note("In matchups, wins/win_pct are from unit_1's point of view.")
    #note("In every other table, wins/win_pct are for armies that contain both items.")
    #note("The win-rate rankings are exploratory. Combinations that mostly appear in big armies look good because the armies are big,"
    #     " and one battle feeds several rows, so rows aren't independent. Check ci_low / ci_high: a wide interval is weak evidence.")

def battle_results(ds: Dataset) -> None:
    title("10. How battles end (survivors and damage)")
    d = _decisive_results(ds)
    draws = ds.battles[ds.battles["complete"] & (ds.battles["winner"] == "draw")]
    line(f"Decisive battles with full results: {len(d)}   Draws with full results: {len(draws)}")
    if d.empty:
        return

    rows = [
        ("Survivors (% of deployed)", "win_survivor_pct", "lose_survivor_pct"),
        ("Survivors (count)", "win_survivors", "lose_survivors"),
        ("Damage dealt", "win_damage_dealt", "lose_damage_dealt"),
        ("Damage dealt / enemy total HP", "win_dmg_vs_enemy_hp", "lose_dmg_vs_enemy_hp"),
        ("Army size", "win_units", "lose_units"),
    ]
    summary = pd.DataFrame([{
        "metric": label,
        "winner_mean": d[w].mean(), "winner_median": d[w].median(),
        "loser_mean": d[l].mean(), "loser_median": d[l].median(),
    } for label, w, l in rows])
    show(summary)

    line()
    line(f"Winner dealt more damage than loser:  {pct((d['win_damage_dealt'] > d['lose_damage_dealt']).sum(), len(d))}")
    line(f"Loser was fully eliminated (0 left):   {pct((d['lose_survivors'] == 0).sum(), len(d))}")
    line(f"Winner kept 50%+ of its army:          {pct((d['win_survivor_pct'] >= 50).sum(), len(d))}")
    subtitle("Winner's survivor % (margin of victory)")
    show(d["win_survivor_pct"].describe(percentiles=[0.25, 0.5, 0.75]).to_frame().T)
    note("'Damage dealt / enemy total HP' above 1.0 means overkill (damage that "
         "exceeded the enemy's total health, or hit already-dying units).")
    note("These columns are OUTCOMES. Do not use them as features to predict the winner.")

def consistency(ds: Dataset) -> None:
    title("11. Repeated matchups (is the game deterministic?)")
    ur = ds.unit_rows.sort_values(["battle_id", "side", "unit_id"]).copy()
    ur["token"] = ur["unit_id"].astype(str) + "x" + ur["count"].astype(str)
    unit_sig = ur.groupby(["battle_id", "side"])["token"].agg(",".join).rename("units_sig")

    if ds.artifact_rows.empty:
        art_sig = pd.Series("", index=unit_sig.index, name="art_sig")
    else:
        ar = ds.artifact_rows.sort_values(["battle_id", "side", "artifact_id"]).copy()
        ar["token"] = ar["artifact_id"].astype(str)
        art_sig = ar.groupby(["battle_id", "side"])["token"].agg(",".join).rename("art_sig")

    sig = pd.concat([unit_sig, art_sig], axis=1).fillna("")
    sig["sig"] = sig["units_sig"] + "|" + sig["art_sig"]
    wide = sig["sig"].unstack("side")
    wide.columns.name = None

    m = wide.join(ds.battles.set_index("battle_id")[["winner"]]).dropna(subset=["A", "B"])
    mirror = int((m["A"] == m["B"]).sum())
    m = m[m["A"] != m["B"]].copy()
    if m.empty:
        line("Not enough data.")
        return

    # Canonical order so 'X vs Y' and 'Y vs X' count as the same matchup.
    swap = m["A"] > m["B"]
    m["matchup"] = (m["B"] + " vs " + m["A"]).where(swap, m["A"] + " vs " + m["B"])
    a_won, b_won = m["winner"] == "A", m["winner"] == "B"
    m["rel"] = "draw"
    m.loc[(a_won & ~swap) | (b_won & swap), "rel"] = "first"
    m.loc[(b_won & ~swap) | (a_won & swap), "rel"] = "second"

    rep = m.groupby("matchup")["rel"].agg(count="count", outcomes="nunique")
    repeated = rep[rep["count"] > 1]
    conflicting = repeated[repeated["outcomes"] > 1]

    line(f"Distinct matchups (identical armies and artifacts, either side): {len(rep)}")
    line(f"Matchups recorded more than once:                               {len(repeated)}")
    line(f"  ...of those, with conflicting outcomes:                       {len(conflicting)}")
    line(f"Mirror matches (identical army on both sides, excluded):        {mirror}")
    if len(repeated):
        line(f"Repeat consistency: {pct(len(repeated) - len(conflicting), len(repeated))}")
    if len(conflicting):
        display_top_n = False
        subtitle("Battle ids with identical setups but different outcomes" + 
                 (f" (up to {TOP_N})" if display_top_n else "")
                 )
        ids = (m[m["matchup"].isin(conflicting.index)].reset_index().groupby("matchup")["battle_id"].agg(list))
        if display_top_n:
            ids = ids.head(TOP_N)
        for battle_ids in ids:
            line(f"battles {battle_ids}")
    note("If the game is deterministic, conflicts mean a data-entry mistake. If it has randomness, conflicts show a ceiling on achievable prediction accuracy.")


def data_quality(ds: Dataset) -> None:
    title("12. Data quality checks")
    b = ds.battles
    d = _decisive_results(ds)
    done = b[b["complete"]]
    draws = done[done["winner"] == "draw"]

    subtitle("Rule: the winner has at least 1 survivor, the loser has 0")
    bad_winner = d[d["win_survivors"] < 1]
    bad_loser = d[d["lose_survivors"] > 0]
    violations = sorted({int(x) for x in bad_winner["battle_id"]} | {int(x) for x in bad_loser["battle_id"]})
    line(f"Decisive battles checked:      {len(d)}")
    line(f"Winner has 0 survivors:        {len(bad_winner)}")
    line(f"Loser still has survivors:     {len(bad_loser)}")
    line(f"Battles violating the rule:    {len(violations)}")
    if violations:
        more = " ..." if len(violations) > 50 else ""
        line(f"Battle ids to review: {violations[:50]}{more}")
    else:
        line("All checked battles follow the rule.")
    note("battles without survivor data (older entries) can't be checked.")

    subtitle("Other checks")
    line(f"Battles missing survivors/damage (legacy or incomplete): {int((~b['complete']).sum())}")
    line(f"Battles where a side has no unit records:                {int(b[['a_units', 'b_units']].isna().any(axis=1).sum())}")
    line(f"Draws where both sides were wiped out:                   {int(((draws['a_survivors'] == 0) & (draws['b_survivors'] == 0)).sum())}")
    line(f"Draws where both sides still have survivors:             {int(((draws['a_survivors'] > 0) & (draws['b_survivors'] > 0)).sum())}")
    line(f"Battles where both sides dealt 0 damage:                 {int(((done['a_damage_dealt'] == 0) & (done['b_damage_dealt'] == 0)).sum())}")
    loser_out_damage_count = int((d['lose_damage_dealt'] > d['win_damage_dealt']).sum())
    line(f"Decisive battles where the loser out-damaged the winner: {loser_out_damage_count} ({100*loser_out_damage_count / len(d) :.2f}%)")
    note("the 'other checks' aren't necessarily errors (timeouts, overkill and "
         "regeneration can all cause them). They are worth a second look if the counts are large.")

def ml_readiness(ds: Dataset) -> None:
    title("13. Machine-learning readiness")
    d = ds.battles[ds.battles["winner"] != "draw"]
    n = len(d)
    seen_units = ds.unit_rows["unit_id"].nunique()
    seen_artifacts = ds.artifact_rows["artifact_id"].nunique()

    line(f"Decisive battles (usable training rows): {n}")
    line(f"After side-swap augmentation:            {2 * n}")
    line(f"Draws (excluded from a win/lose model):  {len(ds.battles) - n}")
    line(f"Unit types seen in battles:              {seen_units} of {len(ds.units)}")
    line(f"Artifacts seen in battles:               {seen_artifacts} of {len(ds.artifacts)}")

    features = seen_units + seen_artifacts
    if features:
        rows_per_features = n / features
        line(f"Battles per raw feature: {rows_per_features:.2f}")
        if rows_per_features < 5:
            note("under ~5 battles per feature is danger zone. Only heavily regularized, simple models would work.")
        elif rows_per_features < 10:
            note("under ~10 battles per feature is thin for per-unit features; expect wide uncertainty.")
        elif rows_per_features < 20:
            note("under ~20 battles per feature is marginal for per-unit features; keep the model simple.")
        elif rows_per_features < 50:
            note("under ~50 battles per feature is workable for per-unit features; comfortable for more complex models.")
        else:
            note("over ~50 battles per feature is satisfactory for per-unit features; flexible models, interactions, small effects, noisy labels, and neural networks.")

    per_unit = ds.unit_rows.groupby("unit_id")["battle_id"].nunique()
    rare = int((per_unit < MIN_SAMPLE).sum())
    line(f"Unit types in fewer than {MIN_SAMPLE} battles: {rare} "
         f"(plus {len(ds.units) - seen_units} never used)")

    h = _heuristic_table(ds)
    if not h.empty:
        best = h.sort_values("accuracy_pct", ascending=False).iloc[0]
        line(f"Best single-metric heuristic: '{best['metric']}' at {best['accuracy_pct']:.2f}% "
             f"over {int(best['compared'])} battles. A model should beat this.")
    note("when you train, keep a battle and its side-swapped copy in the same "
         "cross-validation fold (e.g. GroupKFold on battle id), or the score will be inflated.")

# Entry point
def analyse() -> None:
    conn = get_connection()
    init_db(conn)
    conn.row_factory = None  # pandas wants plain tuples, not sqlite3.Row
    ds = load_dataset(conn)
    conn.close()

    if ds.battles.empty or ds.unit_rows.empty:
        print("No battles recorded yet. Record some battles first.")
        return

    for section in (overview, timeline, outcomes, catalog_usage, army_composition,
                    strength_vs_outcome, unit_performance, artifact_performance,
                    combinations_section, battle_results, consistency,
                    data_quality, ml_readiness):
        section(ds)
    print()