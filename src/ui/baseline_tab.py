import sqlite3

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QButtonGroup, QDoubleSpinBox, QGridLayout, QHBoxLayout, QLabel, QMessageBox,
    QPushButton, QRadioButton, QScrollArea, QVBoxLayout, QWidget,
)

from src.battles import battles
from src.catalog import units
from .army_panel import ChoiceCombo

WINNER_BY_BUTTON_ID = {1: "A", 2: "B", 3: "draw"}

class MatchupRow:
    """ One 1v1 battle: army A (the selected unit) against one unit type as army B """

    def __init__(self, a_unit, b_unit):
        self.a_id, self.b_id = a_unit["id"], b_unit["id"]
        self.text = f"{a_unit['name']}  vs  {b_unit['name']}"
        mirror = "  (mirror)" if self.a_id == self.b_id else ""
        self.label = QLabel(self.text + mirror)

        self.group = QButtonGroup(self.label)
        self.buttons = []
        for button_id, caption in ((1, "A wins"), (2, "B wins"), (3, "Draw")):
            button = QRadioButton(caption)
            self.group.addButton(button, button_id)
            self.buttons.append(button)

        self.damage_a = self._damage_spin()
        self.damage_b = self._damage_spin()

    @staticmethod
    def _damage_spin() -> QDoubleSpinBox:
        spin = QDoubleSpinBox()
        spin.setRange(0, 10_000_000)
        spin.setDecimals(2)
        spin.setFixedWidth(110)
        return spin

    def winner(self):
        """'A', 'B', 'draw', or None while no outcome is picked."""
        return WINNER_BY_BUTTON_ID.get(self.group.checkedId())

    def state(self):
        return self.group.checkedId(), self.damage_a.value(), self.damage_b.value()

    def restore(self, state):
        button_id, damage_a, damage_b = state
        button = self.group.button(button_id)
        if button is not None:
            button.setChecked(True)
        self.damage_a.setValue(damage_a)
        self.damage_b.setValue(damage_b)


class BaselineTab(QWidget):
    saved = Signal()

    def __init__(self, conn):
        super().__init__()
        self.conn = conn
        self.all_units = []
        self.rows: list[MatchupRow] = []
        self.shown_unit_id = None

        # --- unit picker ---
        self.unit_combo = ChoiceCombo()
        self.unit_combo.lineEdit().setPlaceholderText("Select the unit to test...")
        self.unit_combo.currentIndexChanged.connect(self.on_unit_changed)

        picker = QHBoxLayout()
        picker.addWidget(QLabel("Unit (Army A):"))
        picker.addWidget(self.unit_combo, stretch=1)

        self.passes_label = QLabel()

        # --- the N matchup rows ---
        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)

        # --- save ---
        self.save_button = QPushButton("Save all 1v1 battles")
        self.save_button.clicked.connect(self.save)
        self.progress_label = QLabel()
        self.status = QLabel()

        bottom = QHBoxLayout()
        bottom.addWidget(self.save_button)
        bottom.addWidget(self.progress_label)
        bottom.addWidget(self.status, stretch=1)

        layout = QVBoxLayout(self)
        layout.addLayout(picker)
        layout.addWidget(self.passes_label)
        layout.addWidget(self.scroll, stretch=1)
        layout.addLayout(bottom)

        self.reload()

    # ---------- loading ----------
    def reload(self):
        """(Re)load the unit list. Keeps the selected unit and anything already entered."""
        self.all_units = units.list_all(self.conn)
        ids = {u["id"] for u in self.all_units}
        keep = self.shown_unit_id if self.shown_unit_id in ids else None
        snapshot = {row.b_id: row.state() for row in self.rows}
        self.unit_combo.set_choices(self.all_units, keep)
        self.show_unit(keep, snapshot)

    def on_unit_changed(self, _index=None):
        new_id = self.unit_combo.selected_id()
        if new_id == self.shown_unit_id:
            return
        if any(row.winner() is not None for row in self.rows):
            answer = QMessageBox.question(
                self, "Discard entries?",
                "You've entered results for the current unit. Switch and discard them?",
            )
            if answer != QMessageBox.Yes:
                self.unit_combo.blockSignals(True)
                self.unit_combo.setCurrentIndex(self.unit_combo.findData(self.shown_unit_id))
                self.unit_combo.blockSignals(False)
                return
        self.status.clear()
        self.show_unit(new_id)

    def show_unit(self, unit_id, snapshot=None):
        """Build one row per unit type, with the selected unit as Army A in every row."""
        self.shown_unit_id = unit_id
        self.rows = []
        content = QWidget()
        grid = QGridLayout(content)

        if unit_id is None:
            grid.addWidget(QLabel("Select a unit to record its 1v1 baseline matchups."), 0, 0)
            grid.setRowStretch(1, 1)
            self.passes_label.clear()
        else:
            a_unit = next(u for u in self.all_units if u["id"] == unit_id)
            grid.addWidget(QLabel("<b>Matchup</b>"), 0, 0)
            grid.addWidget(QLabel("<b>Winner</b>"), 0, 1, 1, 3)
            grid.addWidget(QLabel("<b>Army A damage</b>"), 0, 4)
            grid.addWidget(QLabel("<b>Army B damage</b>"), 0, 5)

            for r, b_unit in enumerate(self.all_units, start=1):
                row = MatchupRow(a_unit, b_unit)
                grid.addWidget(row.label, r, 0)
                for c, button in enumerate(row.buttons, start=1):
                    grid.addWidget(button, r, c)
                grid.addWidget(row.damage_a, r, 4)
                grid.addWidget(row.damage_b, r, 5)
                row.group.buttonClicked.connect(lambda *_: self.update_progress())
                if snapshot and b_unit["id"] in snapshot:
                    row.restore(snapshot[b_unit["id"]])
                self.rows.append(row)

            grid.setColumnStretch(0, 1)
            grid.setRowStretch(len(self.all_units) + 1, 1)
            passes = battles.baseline_passes(self.conn, unit_id)
            self.passes_label.setText(
                f"Baseline passes already recorded for {a_unit['name']}: {passes}"
            )

        self.scroll.setWidget(content)  # replaces (and deletes) the old content
        self.update_progress()

    def update_progress(self):
        total = len(self.rows)
        done = sum(1 for row in self.rows if row.winner() is not None)
        self.progress_label.setText(f"Recorded {done} of {total} matchups" if total else "")
        self.save_button.setEnabled(total > 0 and done == total)

    # ---------- saving ----------
    @staticmethod
    def _damage_looks_missing(row: MatchupRow) -> bool:
        winner = row.winner()
        if winner == "A":
            return row.damage_a.value() == 0
        if winner == "B":
            return row.damage_b.value() == 0
        return row.damage_a.value() == 0 or row.damage_b.value() == 0  # draw: both units died

    def save(self):
        if self.shown_unit_id is None or not self.rows:
            return
        if any(row.winner() is None for row in self.rows):
            return

        questionable = [row.text for row in self.rows if self._damage_looks_missing(row)]
        if questionable:
            shown = "\n".join(questionable[:8]) + ("\n..." if len(questionable) > 8 else "")
            answer = QMessageBox.question(
                self, "Zero damage?",
                "These matchups have 0 damage for a unit that won or drew:\n\n"
                f"{shown}\n\nSave anyway?",
            )
            if answer != QMessageBox.Yes:
                return

        batch = []
        for row in self.rows:
            winner = row.winner()
            batch.append(dict(
                winner=winner,
                army_a={row.a_id: 1}, army_b={row.b_id: 1},
                # One unit per side: the winner has it left, the loser (or both, in a draw) does not.
                remaining_a={row.a_id: 1} if winner == "A" else {},
                remaining_b={row.b_id: 1} if winner == "B" else {},
                damage_dealt_a={row.a_id: row.damage_a.value()},
                damage_dealt_b={row.b_id: row.damage_b.value()},
            ))
        try:
            battles.record_many(self.conn, batch)
        except (ValueError, sqlite3.Error) as e:
            QMessageBox.critical(self, "Couldn't save battles", str(e))
            return

        name = next(u["name"] for u in self.all_units if u["id"] == self.shown_unit_id)
        self.unit_combo.blockSignals(True)
        self.unit_combo.setCurrentIndex(-1)
        self.unit_combo.lineEdit().clear()
        self.unit_combo.blockSignals(False)
        self.show_unit(None)  # forces a deliberate choice of the next unit
        self.status.setText(f"Saved {len(batch)} battles for {name}.")
        self.saved.emit()