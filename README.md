# Pixel Wars Win Predictor: project context

> **Status: on hold (hibernating).** The user has paused the project. This file is the hand-off for
> picking it up again. Section 12 says where to restart.
>
> **For the assistant reading this in a new conversation:** this file summarises two long previous
> conversations. Treat it as the source of truth for design decisions, but remember that most code
> described here was written by an assistant and only part of it was run by the user (section 8 says
> which). The user has edited files since, so **ask them to paste the current version of any file before
> editing it.** The user pastes snippets and asks for changes. Full-function or full-file replacements
> with exact edit locations work best. They are on Windows and use Visual Studio (git) and VS Code
> (LaTeX notes). Be honest about what has not been executed, and fix tracebacks they paste.

---

## 1. Goal and the game

**Pixel Wars** is a browser auto-battler, played at <https://armorgames.com/pixel-wars-game/19685>.
You assemble armies of units and artifacts, then the armies fight automatically. Surviving units carry
over to the next round, where you can add units, artifacts and upgrades.

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

- Python project (virtual env `.venv`). The tracebacks show the entry point as `main.py` at the repo
  root (older notes said `src/main.py` and `python -m src.main <command>`), so **check how the user
  launches it**. Commands: `record` (default), `catalog`, `analyse`, `train`, `clear-battles`.
- Packages: `PySide6`, `pandas` (plus `numpy`), `scipy` (Mann-Whitney test), `scikit-learn` (plus
  `joblib`). The installed scikit-learn is recent (1.8 or newer: `LogisticRegressionCV` accepts
  `l1_ratios` and `use_legacy_attributes`). `requirements.txt` is made with `pip freeze`.
- SQLite database at `data/pixel_wars.db`. **All DB access goes through `src/api/database.py`.**
- Git via Visual Studio. A feature branch was created from master to restart the modeling work. Whether
  it was merged into master is unknown.
- `.gitignore` should contain: `.venv/`, the database, `data/backups/`, `data/models/`, and LaTeX build
  files (`*.aux *.log *.out *.toc *.fls *.fdb_latexmk *.synctex.gz *.bbl *.blg`).
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
- **The `battles` table stores only the winner and notes.** Army totals (total HP, attack, averages) are
  *not* stored. They are derived at load time in `src/api/loader.py` from `battle_units` joined to the
  current unit catalog.
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
├── main.py                     # command dispatcher + window launcher (see section 2 on its location)
├── data/
│    ├── pixel_wars.db
│    ├── backups/               # timestamped DB copies made by clear-battles
│    └── models/                # saved models: <name>.joblib
├── notes/unit_types.tex        # LaTeX notes template
└── src/
    ├── api/
    │   ├── database.py
    │   └── loader.py           # builds the Dataset, including the a_* / b_* army totals
    ├── catalog/
    │   ├── units.py
    │   ├── artifacts.py
    │   ├── abilities.py
    │   └── unit_abilities.py
    ├── battles/
    │   ├── battles.py
    │   ├── battle_units.py
    │   └── battle_artifacts.py
    ├── ui/
    │   ├── army_panel.py
    │   ├── record_window.py  record_battle_tab.py  baseline_tab.py
    │   ├── win_predictor.py    # live win chance for the Record Battle tab (see section 6)
    │   └── catalog_window.py  units_tab.py  artifacts_tab.py  abilities_tab.py  unit_abilities_tab.py
    ├── analysis/
    │   ├── stats.py
    │   └── analyse.py
    └── modelling/              # spelled "modelling" in the repo
        ├── features.py
        ├── models.py
        ├── evaluate.py
        └── train.py
```

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
    means 0. It validates (`_validate_side`, `_winner_error`) and saves in one transaction. It writes only
    `winner` and `notes` to `battles`, then one row per unit group and per artifact.
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

### Recorder (`RecordWindow`)

**Tab "Record Battle"** (`RecordBattleTab`): two `ArmyPanel`s, a **win chance label** (see below), a
winner picker (placeholder "Select winner..."), a carry-over checkbox ("Carry winner's survivors into
the next round"), notes, Save and Clear.

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
  `keep_survivors()`. `army()` is assumed to return `{unit_id: count}` and `artifact_ids()` a list.

**Win chance label (unfinished).** A `QLabel` between the army panels and the result row, refreshed by
a 400 ms `QTimer` only when the armies changed (`update_prediction`). It calls `WinPredictor` in
`ui/win_predictor.py`, which loads `data/models/<name>.joblib` (`MODEL_PATH` must match a file that
`train.py` actually saved), builds the model's features, and returns P(A wins) **averaged over both
orderings** (`0.5 * (p(A,B) + 1 - p(B,A))`) so it does not depend on which side an army is entered on.
It shows "Win chance: unavailable" (tooltip = the error) until the army totals step works. State at
hibernation: the label appears, but no real percentage has been shown yet.

- The army totals (`total_hp`, `total_atk`, `units`, `atk_x_rate`, `avg_attack_speed`, `avg_speed`) must
  come from the **same code the loader uses**, to avoid training/serving skew. The plan given to the
  user: extract `add_group_columns(bu)` and `pre_battle_side_metrics(bu)` from `load_dataset` in
  `loader.py`, then have `army_totals` call them on a one-army table. Whether the user applied this is
  unknown.
- Features come from one shared definition: `aggregate_features(a, b)` in `features.py`, used by both
  `build_features` and the predictor, plus `composition_features_from_arrays` for composition columns.
- A **round-trip check** (rebuild features of recorded battles through the UI path and compare with the
  training features, expecting a difference near 1e-12) was written but not run. Do it before trusting
  any percentage.
- `WinPredictor.refresh_catalog()` rebuilds the catalog (the loader's names: `unit_id`, `atk`,
  `attack_speed_rank`) and is called from `rebuild()`. It reads rows with `fetchall` because the UI
  connection uses `sqlite3.Row`, which pandas dislikes. Units with no attack speed give NaN features and
  should produce a clear message instead of a prediction.

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

### Catalog editor (`CatalogWindow`, command `catalog`)

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

## 7. Analysis (command `analyse`, terminal output only)

`src/api/loader.py` (the README used to call it `analysis/loader.py`):

- `load_dataset(conn) -> Dataset`; `load_default_dataset()` opens the DB through `database.py`, loads and
  closes. Dataset fields: `units, artifacts, battles, sides, unit_rows, artifact_rows, abilities,
  unit_abilities, excluded`. **The catalogs use the column names `unit_id` / `atk` (units) and
  `artifact_id` (artifacts), not `id`.** `battles` ids are `battle_id`.
- **`SUSPICIOUS_IDS` (a set at the top of loader.py)** excludes those battles from everything.
  `excluded` is the count. Battle ids restart at 1 if all battles are deleted, so reset this set then.
- `battles` has one row per battle with wide `a_*` / `b_*` columns for `PRE_BATTLE_METRICS` (`units,
  types, total_hp, total_atk, atk_x_rate, avg_speed, avg_attack_speed, artifacts`) and `RESULT_METRICS`
  (`survivors, damage_dealt`), plus `d_*` (A minus B) for the pre-battle metrics, `a_win` (1/0, NaN for
  draws) and `complete` (all survivor/damage data present). The wide columns are made with
  `sides.pivot(...)` and an f-string, so searching the source for the literal text `a_total_hp` finds
  nothing. `sides` has one row per (battle, side). `unit_rows` has one row per (battle, side, unit group)
  with unit stats and `result` (win/loss/draw).
- `atk_x_rate = count * attack * attack_speed_rank` where the rank is Very Slow = 1 ... Very Fast = 5
  (`ATTACK_SPEED_RANK`). It is an **ordinal proxy for damage rate, not real attacks per second**.
  Armies with any unit lacking an attack speed get NaN for the attack speed metrics.
- **Result columns are outcomes. Never use them as model features** (leakage).

`analysis/stats.py`: `title, subtitle, line, note, pct, show, wilson` (95% Wilson interval),
`add_win_rate` (win_pct, ci_low, ci_high from wins/losses, draws excluded). `show` accepts a
`float_format_level` argument (the user added it). Output is plain ASCII for Windows terminals. In
`analyse.py`: `TOP_N = 10`, `MIN_SAMPLE = 5` ("low n" flag).

Sections in `analyse.py` (title numbers are typed by hand and may be out of order): overview, timeline,
outcomes and side balance, catalog and usage (mean/median deployed per unit), unit ability catalog
(abilities per unit, units grouped by ability and by number of abilities with mean stats, median attack
speed, battles, total_deployed), army composition, "does the stronger army win?" (single-metric
heuristics and a majority baseline), unit battle performance (includes the **single-unit 1v1 baseline
matchup matrix** with unseen-matchup counts), artifact performance, combinations (pairs, unit+artifact
combos, matchups), how battles end (`battle_results`, section 13, built on `_decisive_results`), repeated
matchups (noise estimate, flag `DISPLAY_TOP_N_CONFLICTS`), data quality (includes the winner/loser
survivor rule with offending battle ids), ML readiness.

- **Artifact survivor check** (inside `battle_results`, recommended to move into a helper function
  `artifact_survivor_check(ds, d, artifact_name)`): compares the winner's survivor % when the *winning*
  army carried a named artifact (the user tried "Necromancer") against when it did not, with a
  Mann-Whitney U test, an effect-size line (chance a with-artifact win keeps more of its army) and an
  `if/elif` message on the p-value. Two bugs fixed along the way: use `match["id"].iloc[0]` (the whole
  row had been used as the id), and do not `return` inside the block (it exits the whole function).
  It should be restricted to real multi-unit battles, because 1v1s have 0% or 100% survivors by
  construction, and a difference does not show causation (confounding by army strength).

**Baseline skew:** about 70% of battles are single-unit 1v1 baseline battles (latest run: 932 of 1440
decisive battles), so only about 500 are "real" multi-unit battles. A "real battle" filter is
`(a_units > 1) & (b_units > 1)`; there is also a helper `real_battle_ids(ds)` with a commented
alternative that excludes only the exact baseline shape (one unit type, one unit per army, no
artifacts). It has been applied to **`unit_performance`** (via the helper) and **`artifact_performance`**
(written inline). **The other sections still include the baseline battles.**

## 8. Modeling (command `train`)

### Status

- **Verified by the user (they ran it and pasted output):** the full pipeline for the logistic regression
  candidates: dataset load, features, cleaning, grouped cross-validation, the real-vs-baseline split,
  calibration table, final fit, coefficients and saving. The earlier "train isn't working" problem was
  never diagnosed. After the restart on the new branch it ran; the cause was not identified.
- **Written by the assistant, only partly run:** composition features, `SymmetricClassifier`, the
  random forest, gradient boosting and SVM candidates, `describe_feature`, the progress output
  (`VERBOSE`). The user began cross-validating the non-linear models (they were very slow), then hit a
  `KeyError: 'B_art_1'` at `random forest [trees_raw]`. The fix (below) was given but **no results for
  any non-linear candidate or composition logistic model were ever shared.**
- **Known fix for the KeyError:** in `feature_sets`, keep both sides of each raw column when either side
  passes the support filter:
  `raw_ids = sorted({c[2:] for c in _cols(df,"A_",s) + _cols(df,"B_",s)})`, then
  `raw = [f"A_{i}" for i in raw_ids] + [f"B_{i}" for i in raw_ids]`. `mirror_X` also raises a clear error
  for an `A_`/`B_` column without a partner.
- Other names to watch: `train.py` must use `fs = feature_sets(data)` and `fs[set_name]` (not the
  `FEATURE_SETS` dict, which only holds the aggregate sets), and needs `import re` and
  `describe_feature`.

### Design

- Only pre-battle features, expressed as A-minus-B differences or log ratios (so swapping sides flips
  signs). Draws are dropped.
- **Aggregate features** (`features.py`, defined once in `aggregate_features`): `log_hp_ratio`,
  `log_atk_ratio`, `log_size_ratio`, `log_atk_rate_ratio` (the old notes said `log_rate_ratio`),
  `attack_speed_diff`, `move_speed_diff`, `artifact_diff`, using `log1p` for totals. Sets: `hp_only`,
  `size_only`, `core`, `core+rates`, `all`.
- **Composition features** (all antisymmetric except the raw per-side columns): `unit_<id>`
  (`log1p(count_A) - log1p(count_B)`), `art_<id>` (A has it minus B has it), `upair_<i>_<j>` (unit i on A
  against unit j on B minus the mirror, using army shares), `apair_`, `avu_<art>_<unit>` (artifact on A
  against an enemy unit), `awu_<art>_<unit>` (artifact with its own units), and raw `A_unit_*`, `B_unit_*`,
  `A_art_*`, `B_art_*` for tree models. `feature_sets(df, min_support=20)` resolves the sets
  `units`, `arts`, `units+arts`, `unit_matchups`, `full_matchups`, `trees_raw` against the columns that
  exist and prunes columns non-zero in fewer than 20 battles. `mirror_X` / `mirror` swap A and B for
  augmentation. `army_vectors` and `artifact_vectors` build per-type count matrices.
- **Models** (`models.py`): ridge logistic regression (`logistic(fit_intercept, wide=False)`,
  `LogisticRegressionCV` on `StandardScaler(with_mean=False)`; `wide=True` searches stronger
  regularisation; `l1_ratios=(0,)` and `use_legacy_attributes=True` silence scikit-learn FutureWarnings).
  No intercept means equal armies give exactly 50% and side-swap augmentation is redundant. The
  non-linear models (`random_forest`, `grad_boost` with monotonic constraints on the HP, attack and
  size ratios, `svm`) are not symmetric, so they are wrapped in **`SymmetricClassifier`**: it trains on
  battles plus their mirror images, averages the score over both orderings, and fits one temperature
  `t_` on out-of-fold scores so the probabilities are calibrated. Its hyperparameters are fixed and
  conservative (about 500 real battles cannot support aggressive tuning).
- **16 candidates** in `CANDIDATES`: base rate; logreg on `hp_only`, `size_only`, `core`, `core+rates`,
  `all`; logreg with intercept on `all`; logreg on `units+arts`, `unit_matchups`, `full_matchups`; random
  forest, gradient boosting and SVM on `all`; random forest and gradient boosting on `trees_raw`; SVM on
  `units+arts`. The coin flip is commented out. `BASELINES`, `REFERENCE` (the base rate) and
  `CHAMPION = "logreg [all]"` (what a new model must beat) are defined there. `VERBOSE = True` prints
  progress.
- **Evaluation** (`evaluate.py`): `StratifiedGroupKFold` repeated 5 x 5, grouped by a canonical matchup
  key (`matchup_groups`) so repeats and mirror images never straddle folds. `oof_predictions` (prints a
  line per fold with time, ETA and the learned temperature when `verbose=True`), `summarise` (log loss,
  spread across repeats, Brier, accuracy for reference), `per_battle_log_loss`, `paired_delta` (paired
  cluster bootstrap that resamples whole matchups), `calibration_table` (with `calibration_gap`; the
  report prints a weighted average gap and a plain-language verdict at 3%, 6% and 10%).
- **`train.py` flow:** load and show the dataset, build features, clean (`dropna` on the aggregate
  features and `matchup`), summarise the training data (this summary block, including the real-vs-baseline
  and artifact tables, is meant to move to `analyse.py`), cross-validate every candidate on the same rows
  and folds, report log loss overall and split by battle type, compare with the base rate (comparing with
  `CHAMPION` is recommended instead), pick the best model, show calibration, refit on all data, print the
  coefficients (top 15, logistic regression only, with readable names), and save with
  `save_model(name, final, features, trained_on, **extra)` to `data/models/<name>.joblib`. The saved
  dictionary holds `name`, `model`, `features`, `trained_on`, plus `cv_log_loss` and `feature_set`.
  Constants: `MIN_ROWS = 50`, `N_SPLITS = 5`, `N_REPEATS = 5`, `SAVE_MODEL`. The train step was being
  split into `train_logreg(...)` and (commented out for now) `train_random_forest`, `train_grad_boost`,
  `train_svm`, with a shared data preparation step passing the same rows, labels, groups and real-battle
  mask (`is_real`) to every model.

### Results (user's runs, 1440 decisive battles, 983 matchup groups)

| Model (features) | Log loss, all battles | Real multi-unit (508) | Baseline 1v1 (932) |
|---|---|---|---|
| base rate | 0.6633 | 0.5449 | 0.7279 |
| logreg [hp_only] | 0.5649 | 0.6349 | 0.5268 |
| logreg [size_only] | 0.6443 | 0.5546 | 0.6931 |
| logreg [core] | 0.4222 | 0.5402 | 0.3578 |
| logreg [core+rates] | 0.4062 | 0.5330 | 0.3370 |
| logreg [all] | 0.3575 | 0.4013 | 0.3337 |
| logreg+intercept [all] | 0.3569 | 0.3960 | 0.3355 |

(The split columns come from a slightly earlier run. All models beat the base rate with bootstrap
intervals away from zero. The coin flip scores 0.6931.)

Things the data showed:

- **A wins 62.15% overall, but 86.02% of real battles and only 49.14% of 1v1 baseline battles.** The game
  has no side advantage, so the 86% reflects how battles were recorded (A tends to be the army the user
  expected to win, and A has more HP in only 65.9% of real battles). A constant "A wins" guess scores 0.4045
  on real battles.
- Therefore **the fair yardstick for a symmetric model on real battles is a coin flip (0.6931)**, where
  `logreg [all]` scores 0.4013. The intercept model's small edge (0.3960) most likely exploits the
  recording habit and will not transfer to new matchups. **Save a symmetric (no-intercept) model.** The
  selection rule picks the lowest overall log loss and chose the intercept model by 0.0006, inside the
  noise.
- Artifacts, real battles: A has the artifact edge in 324 battles (A wins 89.8%), equal in 164 (84.8%),
  behind in 20 (35.0%). So artifacts seem to matter, but they are not the main source of the A bias.
  `artifact_diff` takes log loss on real battles from 0.533 to 0.401, which should be treated with caution
  (possible confounding) until tested with deliberately randomised battles.
- Coefficients: `log_atk_ratio` (+3.77 raw) and `log_atk_rate_ratio` (-2.26) are strongly correlated and
  offset each other. Read only their combined effect. `log_size_ratio` is largely redundant with total HP
  and attack. Chosen `C` was 12.3 early on and 4.3 with more data (weak shrinkage, see issues below).
- Calibration is decent: most bins within a few points. The 0.4-0.5, 0.5-0.6 and 0.1-0.2 bins were off by
  7 to 10 points but hold only 65 to 92 battles each, which is within noise.

### Known issues and cautions

- **The inner CV in `LogisticRegressionCV` is not grouped**, so near-duplicate matchups leak into its
  splits and favour weak regularisation. Reported scores are slightly optimistic. Fix with precomputed
  grouped inner splits, or nested cross-validation (and note that picking the best of 16 candidates on the
  same CV that reports its score adds a little more optimism).
- `matchup_groups` includes exact counts (10 vs 11 of a unit is a different matchup). For flexible models
  and pair features, group by the *set of unit types* per side instead.
- **Mirror (unit against itself) 1v1 rows give all-zero features**, so symmetric models predict exactly
  0.5 for them. The accuracy column skips exact-0.5 predictions, so it is not comparable across models.
- **The overall score is flattered by easy 1v1s.** Judge models on real-battle log loss, and compare new
  models to `CHAMPION` with `paired_delta`.
- The wide composition sets can have more columns than rows (`full_matchups` up to about 1,800 columns
  against 1,440 battles). Expect little gain; stronger regularisation (`wide=True`) and the support filter
  are essential.
- Do not average a random forest, gradient boosting and SVM into an ensemble just yet. They see the same
  aggregate totals, so averaging repeats the same blind spot, and their probabilities need calibration.
  The cheap test is to average the out-of-fold predictions already in `preds_by_name` and use
  `paired_delta` against `CHAMPION`. Recommended order: logistic on composition features first, then one
  tree model (gradient boosting) with augmentation and calibration, then ensemble only if both help.
- Selection currently still reads the final-model section for logistic regression only. For other winners
  it must skip the coefficient table, and loading a saved wrapped model needs `src.modelling.models`
  importable.
- Models were never compared on random side assignment. **When recording new battles, assign sides by a
  coin flip** and record some close matchups. More varied real battles help more than any model change.
- Guidance already given: expect modest gains over baseline; judge by log loss and calibration; more data
  cannot remove the randomness; use survivor fraction as an auxiliary target; consider a small
  high-replication evaluation set; keep rows-per-feature conservative with noisy labels.

### Remaining upgrade ladder (one rung at a time)

Kept only if repeated grouped CV improves on `CHAMPION` by more than the noise: (1) composition features
with ridge logistic regression, antisymmetric pair features (built, not yet evaluated), (2) low-rank
factorised pair interactions, (3) per-artifact flags (built), (4) shallow gradient boosting with monotonic
constraints and side-swap augmentation (built, not evaluated), (5) abilities as army-level features (for
example, number of deployed units with ability X) and unit-role clustering.

## 9. Dataset snapshot (from the last run, may have changed)

1469 battles (1440 decisive, draws excluded), of which 932 are single-unit baseline and 508 are real
multi-unit decisive battles, 31 unit types, 19 artifacts, abilities assigned to units. All unit types have
had one full 1v1 baseline pass. A `clear-battles` command exists (typed `DELETE` confirmation plus an
automatic backup in `data/backups/`) but it is unknown whether it was run.

## 10. Conventions and gotchas

- Units are referenced by id everywhere. Side "A"/"B" is arbitrary in the game, so models should be
  symmetric, but see the recording bias above.
- Win rates exclude draws and carry Wilson intervals. Unit and artifact win rates are confounded and
  descriptive only.
- Editing a unit's stats rewrites history for old battles (the loader joins current stats), and a saved
  model's weights then no longer match the data it was trained on. Retrain after changing stats. A
  possible fix is snapshotting stats per battle. Not built.
- **Training/serving skew:** a saved model needs the same feature columns, definitions and order at
  prediction time. Keep one definition (`aggregate_features`, `composition_features_from_arrays`, the
  loader's totals). The saved file knows the column names but not what they mean. Worth also saving a
  feature-definition version, the scikit-learn version, and the training range of army sizes.
- Two processes (recorder and catalog) are safe for SQLite at this scale.
- pandas pitfalls handled so far: all-NaN sums become 0 unless `min_count=1`, numpy ints print as
  `np.int64(..)`, empty tables come back as `object` dtype (cast after reading), and `row_factory =
  sqlite3.Row` breaks `pd.read_sql` (use `fetchall` or set it to `None`).
- Name clash that caused a bug: a DataFrame called `real` (real battles) and a boolean mask called `real`.
  The mask is now `is_real`.
- Replacing a function: show the whole function plus where it goes. The user pastes real code back.

## 11. Backlog and open items

1. **Finish the live win chance label:** extract the loader's totals code, fill in `army_totals`, set
   `MODEL_PATH`, run the round-trip check, add a "Reload model" button, warn on units not seen in
   training and on very large armies.
2. **Run and read the non-linear and composition candidates** (apply the `trees_raw` fix first). Keep
   the symmetric champion unless a model is clearly better on real-battle log loss.
3. Change model selection: prefer symmetric models and require a "better" verdict against `CHAMPION`
   before saving something else. Group the inner CV, and consider nested cross-validation.
4. Apply the "real battle" filter (or an explicit baseline vs real split) to the remaining analysis
   sections: usage, army composition, stronger-army-wins, combinations, how battles end, ML readiness.
5. Move the training-data summary block from `train.py` into `analyse.py`.
6. Record more real battles with randomised sides and some close matchups, including controlled pairs that
   differ by one unit or artifact.
7. A **History tab** to view, edit and delete saved battles has been offered repeatedly but not built.
   Today the only tools are `SUSPICIOUS_IDS` and `clear-battles`.
8. A win-rate matrix for the single-unit baseline cells and a noise-ceiling section in the analysis
   (offered, not built).
9. Ability-based features for the model, and optional unit-role clustering features.
10. The undefined draw rule for the winner/survivor validation, if the user wants one.
11. Merge the feature branch into master and push (instructions were given).
12. Filling in the LaTeX notes per unit type.

## 12. How to resume

1. Ask the user to paste the files they want changed. They have edited files since the code was
   written, and some of the pieces above were never run.
2. Run `train` again and look at the cross-validation table, the by-battle-type table and the delta table
   first. If a non-linear model fails, ask for the traceback.
3. Then follow backlog item 1 (the win chance label) or item 2 (the composition candidates), whichever
   the user wants. Both are small steps from where the work stopped.
