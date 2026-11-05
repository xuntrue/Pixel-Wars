# Pixel Wars Win Predictor: project context

> **For the assistant reading this in a new conversation:** this file summarises a long previous
> conversation. Treat it as the source of truth for design decisions, but the code described here was
> written by an assistant that never ran it, and the user has edited files since (for example they
> renamed `winner_error` to `_winner_error`). **Ask the user to paste the current version of any file
> before editing it.** The user pastes snippets and asks for changes. Full-function or full-file
> replacements with exact edit locations work best. They are on Windows and use Visual Studio (git)
> and VS Code (LaTeX notes). Be honest that code has not been executed, and fix tracebacks they paste.

---

## 1. Goal and the game

**Pixel Wars** is a browser auto-battler. You assemble armies of units and artifacts, then the armies fight automatically.
Surviving units carry over to the next round, where you can add units, artifacts and upgrades.

**Project goal:** Given two teams, predict the probability that each team wins.

- An **army** is X units of each of N unit types. There is no limit on units per type, so the space of
  possible compositions is effectively infinite.
- **Artifacts** are boolean flags per army (the army has it or not).
- A **unit type** has: `damage`, `speed` (MOVEMENT speed), `health`, `attack_speed` (a category, see
  below), and 0 to N **abilities** (secret abilities).
- `attack_speed` is one of `Very Slow, Slow, Medium, Fast, Very Fast` and is different from `speed`.
- **Outcomes are random.** Simulating the identical matchup repeatedly gives different winners, damage
  and survivors. So the model target is a win *probability*, accuracy has a ceiling below 100%, and
  evaluation should use log loss, Brier score and calibration, not accuracy.

The user is building this as a data analysis / ML portfolio project (GitHub). They are comfortable with
computer science and are learning data analysis and ML, so explanations should be practical and clear.

## 2. Environment and workflow

- Python project, run from the repo root with `python -m src.main <command>` (virtual env `.venv`).
- Packages: `PySide6`, `pandas` (plus `numpy`), `scikit-learn` (plus `joblib`). `requirements.txt` is
  made with `pip freeze`.
- SQLite database at `data/pixel_wars.db`. **All DB access goes through `src/api/database.py`.**
- Git via Visual Studio. The user created a feature branch from master to restart the modeling work and
  asked how to merge it into master. Whether that merge has happened is unknown.
- `.gitignore` should contain: `.venv/`, the database, `data/backups/`, and LaTeX build files
  (`*.aux *.log *.out *.toc *.fls *.fdb_latexmk *.synctex.gz *.bbl *.blg`).
- LaTeX notes are written in VS Code (LaTeX Workshop). A template `notes/unit_types.tex` exists with
  macros `\unit{Name}{Damage}{Move speed}{Attack speed}{Health}`, `\unitdata{...}` (data summary table)
  and `\unitref{Name}` (cross links). Special characters must be escaped (`\%`, `\&`, `\_`, `\#`).

## 3. Data model (SQLite)

| Table | Columns / rules |
|---|---|
| `units` | `id`, `name` UNIQUE, `damage` REAL >= 0, `speed` REAL >= 0, `attack_speed` TEXT nullable CHECK in the 5 values, `health` REAL > 0 |
| `artifacts` | `id`, `name` UNIQUE |
| `abilities` | `id`, `name` UNIQUE |
| `unit_abilities` | `unit_id` FK cascade, `ability_id` FK cascade, PK (unit_id, ability_id). Many-to-many, a unit has 0 to N abilities |
| `battles` | `id`, `timestamp` (UTC, default `datetime('now')`), `winner` in ('A','B','draw'), `notes`. **No `carry_over` column** (removed) |
| `battle_units` | `battle_id` FK cascade, `side` 'A'/'B', `unit_id` FK RESTRICT, `count` > 0, `remaining` >= 0 (nullable), `damage_dealt` >= 0 (nullable), PK (battle_id, side, unit_id) |
| `battle_artifacts` | `battle_id` FK cascade, `side`, `artifact_id` FK RESTRICT, PK (battle_id, side, artifact_id) |

Key facts:

- **Survivors (`remaining`) and `damage_dealt` are stored per unit type within a side** (a "unit group"),
  not per army and not per individual unit. Total damage divided by attack does not divide evenly, so
  per-unit damage cannot be verified.
- Nullable `remaining` / `damage_dealt` / `attack_speed` mean "unknown" (legacy rows). Never treat NULL as 0.
- `ON DELETE RESTRICT` on units and artifacts used in battles protects training data. Deleting such a
  unit raises `sqlite3.IntegrityError`.
- **Validation rule for decisive battles:** the winning army has at least 1 survivor and the losing army
  has 0. Draws are not checked by this rule. (In the 1v1 baseline tab a draw means both units died.)
- `database.py` contains: `DATA_DIR`, `DB_PATH`, `ATTACK_SPEEDS` (tuple, slowest to fastest, single source
  of truth), `SCHEMA` (an f-string, so no stray `{}` in the SQL), `MIGRATIONS` (list of
  `(table, column, definition)` added via `ALTER TABLE ADD COLUMN` if missing), `OBSOLETE_COLUMNS`
  (`battles.damage_a`, `battles.damage_b`, `battles.carry_over`, dropped with `ALTER TABLE DROP COLUMN`,
  silently skipped on SQLite older than 3.35), `get_connection()` (enables `PRAGMA foreign_keys`, sets
  `row_factory = sqlite3.Row`), `init_db()` (runs on every app start: creates tables, applies migrations,
  drops obsolete columns).

## 4. Repository layout
```
├── data/
│    ├── pixel_wars.db
│    └── backups/            # timestamped DB copies made by clear-battles
├── models/
├── notes/unit_types.tex        # LaTeX notes template
└── src/
    ├── main.py  # command dispatcher + window launcher + clear_all_battles()
    ├── api/
    │   ├── database.py 
    │   └── loader.py
    ├── catalog/
    │   ├── units.py
    │   ├── artifacts.py
    │   ├── abilities.py 
    │   └── unit_abilities.py
    ├── battles/
    │   ├──battles.py
    │   ├──battle_units.py
    │   └──battle_artifacts.py
    ├── ui/
    │   ├── army_panel.py
    │   ├── record_window.py record_battle_tab.py  baseline_tab.py
    │   └── catalog_window.py units_tab.py  artifacts_tab.py  abilities_tab.py  unit_abilities_tab.py 
    ├── analysis/
    │   ├── stats.py
    │   └── analyse.py
    └── modeling/
        ├── features.py
        ├── models.py
        ├── evaluate.py
        └── train.py
```

`main.py` commands (default `record`): `record`, `catalog`, `analyse`, `train`, `clear-battles`.
Example: `python -m src.main analyse`.

## 5. Backend API

Every function takes `conn` as its first parameter.

- `catalog/units.py`: `create(conn, name, damage, speed, health, attack_speed)`, `get`, `list_all`
  (ordered by name), `update(conn, unit_id, name, damage, speed, health, attack_speed)`, `delete`,
  `battle_count(conn, unit_id)`. Attack speed is validated against `ATTACK_SPEEDS` (`ValueError`).
- `catalog/artifacts.py`: `create`, `list_all`, `rename`, `delete`.
- `catalog/abilities.py`: `create`, `list_all`, `rename`, `unit_count`, `delete` (also removes links).
- `catalog/unit_abilities.py`: `ability_ids_for_unit`, `set_for_unit` (replace the set), `names_by_unit`.
- `battles/battles.py`:
  - `record(conn, winner, army_a, army_b, remaining_a, remaining_b, damage_dealt_a, damage_dealt_b,
    artifacts_a=(), artifacts_b=(), notes=None) -> battle_id`. Dict arguments map `unit_id -> value`
    (`army` = count deployed, `remaining` = survivors, `damage_dealt` = group damage). A missing key
    means 0. It validates (`_validate_side`, `_winner_error`) and saves in one transaction.
  - `record_many(conn, batch)` saves a list of kwarg dicts all-or-nothing (validates everything first).
  - Internals: `_prepare` (validation), `_insert` (no transaction handling).
  - `list_all`, `delete`, `count`, `delete_all` (clears battles and children, keeps units/artifacts),
    `get_details`, `format_details` (readable per-battle dump), `baseline_passes(conn, unit_id)`
    (number of mirror 1v1 battles of a unit, with no artifacts, which equals the passes recorded).

## 6. The two PySide6 apps

Both are separate processes. Edits made in one are picked up by the other after pressing **F5** in the
recorder ("Reload catalog"). Common UI patterns: `refresh()` reloads from the DB after every write,
tabs emit a `changed` / `saved` Signal, unit rows store DB ids in `Qt.UserRole`, and signals that carry
a `checked` bool are connected through `lambda: ...` when the target has default arguments.

### Recorder (`RecordWindow`, `python -m src.main record`)

**Tab "Record Battle"** (`RecordBattleTab`): two `ArmyPanel`s, a winner picker (placeholder
"Select winner..."), a carry-over checkbox ("Carry winner's survivors into the next round"), notes,
Save and Clear.

- Save checks: winner chosen, both armies non-empty, `battles._winner_error(...)` passes, then
  `battles.record(...)`.
- If the carry-over box is ticked and the winner is A or B: the **losing army is cleared**, and the
  winner's army is reduced to its survivors (`ArmyPanel.keep_survivors()`: unit types with 0 survivors
  are removed, deployed becomes the survivor count, survivors and damage reset, artifacts stay), the
  winner picker and notes reset, and the checkbox stays ticked so a chain of rounds can be recorded.
  Untick it for the final round. Draws, or the box unticked, call `clear_form()` (everything resets).
- `ArmyPanel` holds dynamic records. Unit records: searchable type dropdown (`ChoiceCombo`), deployed
  (min 1), survivors (max = deployed), damage dealt, delete button on every record but the first,
  "Add Unit" disappears when all types are used, and a type can appear only once per army (dropdowns
  hide types used elsewhere in the army). Artifact records: dropdown + delete, "Add Artifact". Methods:
  `rebuild()`, `army()`, `remaining()`, `damage_by_unit()`, `artifact_ids()`, `clear()`,
  `keep_survivors()`.

**Tab "1v1 Baseline"** (`BaselineTab`): records the base case of single-unit matchups.

- Pick a unit (Army A). One row per unit type in the catalog (including the unit itself as the mirror).
  Each row has outcome radio buttons (A wins / B wins / Draw) and damage dealt for A and for B.
- Survivors are derived: the winner has 1, the loser 0, a draw gives 0 for both. No artifacts.
- Save is enabled only when every row has an outcome, and saves all N battles atomically via
  `record_many`. A soft warning appears if a winner (or either unit in a draw) has 0 damage.
- It shows "Baseline passes already recorded for <unit>". After saving, the selection is cleared.
- One pass per unit gives 2 records per unordered matchup (once from each side, sides swapped) and 1 per
  mirror matchup. This is intended. Over-repeating a unit is the user's responsibility and is visible
  in the analysis. **The user has recorded one pass for every unit.**

### Catalog editor (`CatalogWindow`, `python -m src.main catalog`)

- **Units tab:** add, edit (double-click or button), delete. Fields: name, damage, "Move speed" (DB
  column `speed`), attack speed (required dropdown), health. Legacy units with NULL attack speed show a
  red "(not set)" and a banner counts them. Changing stats of a unit that appears in recorded battles
  asks for confirmation (filling in a blank attack speed does not). Deleting a unit used in battles is
  refused.
- **Artifacts tab and Abilities tab:** list, add, rename (double-click), delete. Deleting an ability
  warns how many units will lose it.
- **Unit Abilities tab:** left, a unit table with an abilities summary and a filter box. Right, a
  checklist of all abilities for the selected unit, saved immediately on tick/untick. It reloads when
  units or abilities change.

## 7. Analysis (`python -m src.main analyse`, terminal output only)

`analysis/loader.py`:

- `load_dataset(conn) -> Dataset`; `load_default_dataset()` opens the DB through `database.py`, loads and
  closes. Dataset fields: `units, artifacts, battles, sides, unit_rows, artifact_rows, abilities,
  unit_abilities, excluded`.
- **`SUSPICIOUS_IDS` (a set at the top of loader.py)** excludes those battles from everything.
  `excluded` is the count. Battle ids restart at 1 if all battles are deleted, so reset this set then.
- `battles` has one row per battle with wide `a_*` / `b_*` columns for `PRE_BATTLE_METRICS` (`units,
  types, total_hp, total_atk, atk_x_rate, avg_speed, avg_attack_speed, artifacts`) and `RESULT_METRICS`
  (`survivors, damage_dealt`), plus `d_*` (A minus B) for the pre-battle metrics, `a_win` (1/0, NaN for
  draws) and `complete` (all survivor/damage data present). `sides` has one row per (battle, side).
  `unit_rows` has one row per (battle, side, unit group) with unit stats and `result` (win/loss/draw).
- `atk_x_rate = count * attack * attack_speed_rank` where the rank is Very Slow = 1 ... Very Fast = 5
  (`ATTACK_SPEED_RANK`). It is an **ordinal proxy for damage rate, not real attacks per second**.
  Armies with any unit lacking an attack speed get NaN for the attack speed metrics.
- **Result columns are outcomes. Never use them as model features** (leakage).

`analysis/stats.py`: `title, subtitle, line, note, pct, show, wilson` (95% Wilson interval),
`add_win_rate` (win_pct, ci_low, ci_high from wins/losses, draws excluded). Output is plain ASCII for
Windows terminals. In `analyse.py`: `TOP_N = 10`, `MIN_SAMPLE = 5` ("low n" flag).

Sections in `analyse.py` (title numbers are typed by hand and may be out of order): overview, timeline,
outcomes and side balance, catalog and usage (mean/median deployed per unit), unit ability catalog
(abilities per unit, units grouped by ability and by number of abilities with mean stats, median attack
speed, battles, total_deployed), army composition, "does the stronger army win?" (single-metric
heuristics and a majority baseline), unit battle performance (includes the **single-unit 1v1 baseline
matchup matrix** with unseen-matchup counts), artifact performance, combinations (pairs, unit+artifact
combos, matchups), how battles end, repeated matchups (noise estimate, flag
`DISPLAY_TOP_N_CONFLICTS`), data quality (includes the winner/loser survivor rule with offending battle
ids), ML readiness.

**Baseline skew:** in the last run, 961 of 1362 battles (70.6%) qualified as single-unit 1v1 baseline
battles, so only about 400 are "real" multi-unit battles. A "real battle" filter is
`(a_units > 1) & (b_units > 1)`; there is also a helper `real_battle_ids(ds)` with a commented
alternative that excludes only the exact baseline shape (one unit type, one unit per army, no
artifacts). It has been applied to **`unit_performance`** (via the helper) and **`artifact_performance`**
(written inline). **The other sections still include the baseline battles.**

## 8. Modeling (`python -m src.main train`)

**Status: unresolved.** The user ran it, reported "this isn't working" (without sharing the error), and
started a fresh branch. Treat the design below as intent, not verified code.

- Design: only pre-battle features, expressed as A-minus-B differences or log ratios (so swapping sides
  flips signs). Features (`features.py`): `log_hp_ratio`, `log_atk_ratio`, `log_size_ratio`,
  `log_rate_ratio`, `attack_speed_diff`, `move_speed_diff`, `artifact_diff`, using `log1p` for totals.
  Feature sets: `hp_only`, `size_only`, `core`, `core+rates`, `all`. Draws are dropped.
- Model: ridge logistic regression (`LogisticRegressionCV`) on `StandardScaler(with_mean=False)` with no
  intercept (equal armies give exactly 50%, and side-swap augmentation is redundant). Also a variant
  with an intercept to detect side bias. Baselines: coin flip and base rate.
- Evaluation (`evaluate.py`): `StratifiedGroupKFold` repeated, grouped by a canonical matchup key
  (`matchup_groups`) so repeats and mirror images never straddle folds. Metrics are log loss, Brier,
  accuracy (informational), a paired cluster-bootstrap comparison against the base rate, and a
  calibration table. `train.py` saves the best model to `data/models/win_model.joblib` only if it beats
  the baselines.
- Planned upgrade ladder, one rung at a time and kept only if repeated grouped CV improves by more than
  the noise: (1) unit-role clusters from stats, (2) per-artifact flags with L1, (3) raw per-type count
  differences with ridge (try `log1p(count)`), (4) shallow gradient boosting with monotonic constraints,
  trained with side-swap augmentation and symmetrised predictions `(P(A,B) + 1 - P(B,A)) / 2`.
  Abilities will become army-level features (for example, number of deployed units with ability X).
- Guidance already given: expect modest gains over baseline; judge by log loss (coin flip = 0.6931) and
  calibration; more data cannot remove the randomness; use the continuous outcomes (survivor fraction)
  as an auxiliary or alternative target; consider a small high-replication evaluation set; keep
  rows-per-feature conservative with noisy labels.

## 9. Dataset snapshot (from the last analysis run, may have changed)

About 1362 battles (961 are single-unit baseline), 31 unit types, 19 artifacts, abilities assigned to
units. All unit types have had one full 1v1 baseline pass. A `clear-battles` command exists (typed
`DELETE` confirmation plus an automatic backup in `data/backups/`) but it is unknown whether it was run.

## 10. Conventions and gotchas

- Units are referenced by id everywhere. Side "A"/"B" is arbitrary, so models should be symmetric.
- Win rates exclude draws and carry Wilson intervals. Unit and artifact win rates are confounded and
  descriptive only.
- Editing a unit's stats rewrites history for old battles (the analysis joins current stats). A possible
  fix is snapshotting stats per battle if game patches matter. Not built.
- Two processes (recorder and catalog) are safe for SQLite at this scale.
- pandas pitfalls handled so far: all-NaN sums become 0 unless `min_count=1`, numpy ints print as
  `np.int64(..)`, empty tables come back as `object` dtype (cast after reading).
- Replacing a function: show the whole function plus where it goes. The user pastes real code back.

## 11. Backlog and open items

1. Apply the "real battle" filter (or an explicit baseline vs real split) to the remaining analysis
   sections: usage, army composition, stronger-army-wins, combinations, how battles end, ML readiness.
   Offered an optional split of usage columns into baseline and real battles.
2. Get the modeling pipeline working and decide how training treats the 70% baseline rows (weighting, or
   scoring baseline and real battles separately).
3. A **History tab** to view, edit and delete saved battles has been offered repeatedly but not built.
   Today the only tools are `SUSPICIOUS_IDS` and `clear-battles`.
4. A win-rate matrix for the single-unit baseline cells (offered, not built), and a noise-ceiling section
   in the analysis (offered, not built).
5. Ability-based features for the model, and the optional unit-role clustering features.
6. The undefined draw rule for the winner/survivor validation, if the user wants one.
7. Merge the feature branch into master and push (instructions were given).
8. Filling in the LaTeX notes per unit type.
