import joblib
import numpy as np
import pandas as pd

from src.api.database import DATA_DIR
from src.api.loader import ATTACK_SPEED_RANK, add_group_columns, pre_battle_side_metrics
from src.modelling.features import aggregate_features, composition_features_from_arrays

MODEL_PATH = DATA_DIR / "models" / "logreg_units_arts.joblib"   # which saved model the UI uses; change to switch

def army_totals(army: dict, artifact_ids: list, units: pd.DataFrame) -> dict:
    """ Totals for one army, computed by the same loader functions that build ds.battles """
    army = {u: c for u, c in army.items() if c > 0}
    rows = pd.DataFrame({"battle_id": 0, "side": "X", "unit_id": list(army), "count": list(army.values())})
    rows = rows.merge(units[["unit_id", "atk", "speed", "health", "attack_speed_rank"]], on="unit_id", how="left")
    m = pre_battle_side_metrics(add_group_columns(rows)).iloc[0]
    return {
            "total_hp": m["total_hp"],
            "total_atk": m["total_atk"],
            "units": m["units"],
            "atk_x_rate": m["atk_x_rate"],
            "avg_attack_speed": m["avg_attack_speed"],
            "avg_speed": m["avg_speed"],
            "artifacts": len(artifact_ids)
        }

class WinPredictor:
    def __init__(self, conn):
        self.conn = conn
        self.model = None
        self.features: list[str] = []
        self.info = "No saved model"
        self.refresh_catalog()
        self.load()

    def load(self) -> None:
        try:
            saved = joblib.load(MODEL_PATH)
        except FileNotFoundError:
            self.model, self.info = None, f"No saved model at {MODEL_PATH.name}"
            return
        self.model, self.features = saved["model"], saved["features"]
        self.info = f"{saved.get('name', MODEL_PATH.stem)}, trained on {saved['trained_on']} battles"

    def refresh_catalog(self) -> None:
        """ Re-read the unit and artifact catalogs (call after they are edited) """
        rows = self.conn.execute("SELECT id, name, damage, speed, attack_speed, health FROM units").fetchall()
        self.units = pd.DataFrame([tuple(r) for r in rows],
                                  columns=["unit_id", "name", "atk", "speed", "attack_speed", "health"])
        self.units["attack_speed_rank"] = self.units["attack_speed"].map(ATTACK_SPEED_RANK)
        self.unit_ids = sorted(self.units["unit_id"])
        self.art_ids = sorted(r[0] for r in self.conn.execute("SELECT id FROM artifacts").fetchall())

    def _vec(self, ids: list, chosen: dict) -> np.ndarray:
        return np.array([[chosen.get(i, 0) for i in ids]], dtype=float)

    def features_for(self, army_a, art_a, army_b, art_b) -> pd.DataFrame:
        """ One row of model inputs for 'A vs B'. Same code path as training """
        agg = aggregate_features(army_totals(army_a, art_a, self.units), army_totals(army_b, art_b, self.units))
        comp = composition_features_from_arrays(
            self._vec(self.unit_ids, army_a), self._vec(self.unit_ids, army_b),
            self._vec(self.art_ids, {i: 1 for i in art_a}), self._vec(self.art_ids, {i: 1 for i in art_b}),
            self.unit_ids, self.art_ids)
        return pd.concat([pd.DataFrame([agg]), comp], axis=1)

    def _p(self, army_a, art_a, army_b, art_b) -> float:
        X = self.features_for(army_a, art_a, army_b, art_b)
        if X[self.features].isna().any(axis=None):
            raise ValueError("A unit in this matchup has no attack speed set")
        missing = [c for c in self.features if c not in X.columns]
        if missing:
            raise KeyError(f"Model needs columns the UI cannot build: {missing[:5]}")
        return float(self.model.predict_proba(X[self.features])[0, 1])

    def predict(self, army_a, art_a, army_b, art_b):
        """ P(army A wins), the same whichever side each army is entered on. None if not possible """
        if self.model is None or not army_a or not army_b:
            return None
        return 0.5 * (self._p(army_a, art_a, army_b, art_b) + 1 - self._p(army_b, art_b, army_a, art_a))