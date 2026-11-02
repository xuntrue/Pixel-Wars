import sqlite3

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QHBoxLayout, QLabel, QLineEdit, QMessageBox,
    QPushButton, QVBoxLayout, QWidget,
)

from src.battles import battles
from .army_panel import ArmyPanel

class RecordBattleTab(QWidget):
    saved = Signal()  # emitted after a battle is saved (the History tab will listen)

    def __init__(self, conn):
        super().__init__()
        self.conn = conn

        self.panel_a = ArmyPanel("Army A", conn)
        self.panel_b = ArmyPanel("Army B", conn)
        panels = QHBoxLayout()
        panels.addWidget(self.panel_a)
        panels.addWidget(self.panel_b)

        # --- result row ---
        self.winner_box = QComboBox()
        self.winner_box.addItem("Select winner...", None)
        self.winner_box.addItem("Army A wins", "A")
        self.winner_box.addItem("Army B wins", "B")
        self.winner_box.addItem("Draw", "draw")

        self.carry_check = QCheckBox("Carry winner's survivors")
        self.notes_edit = QLineEdit()
        self.notes_edit.setPlaceholderText("Notes (optional)")

        result_row = QHBoxLayout()
        result_row.addWidget(self.winner_box)
        result_row.addWidget(self.carry_check)
        result_row.addWidget(self.notes_edit, stretch=1)

        # --- buttons + status ---
        save_button = QPushButton("Save battle")
        save_button.clicked.connect(self.save)
        clear_button = QPushButton("Clear form")
        clear_button.clicked.connect(self.clear_form)
        self.status = QLabel("")

        button_row = QHBoxLayout()
        button_row.addWidget(save_button)
        button_row.addWidget(clear_button)
        button_row.addWidget(self.status, stretch=1)

        # --- assemble ---
        layout = QVBoxLayout(self)
        layout.addLayout(panels, stretch=1)
        layout.addLayout(result_row)
        layout.addLayout(button_row)

    def rebuild(self):
        """Connected to the Units/Artifacts tabs' `changed` signals."""
        self.panel_a.rebuild()
        self.panel_b.rebuild()

    def clear_form(self):
        self.panel_a.clear()
        self.panel_b.clear()
        self.winner_box.setCurrentIndex(0)
        self.carry_check.setChecked(False)
        self.notes_edit.clear()

    def carry_winner_forward(self, winner: str):
        """ Clear the losing army, keep the winner's survivors and artifacts """
        winner_panel, loser_panel = ((self.panel_a, self.panel_b) if winner == "A" else (self.panel_b, self.panel_a))
        loser_panel.clear()
        winner_panel.keep_survivors()

        self.winner_box.setCurrentIndex(0)
        self.notes_edit.clear()
        # The checkbox stays ticked so a chain of rounds can be recorded in a row

    def save(self):
        winner = self.winner_box.currentData()
        if winner is None:
            QMessageBox.warning(self, "Missing winner", "Select who won the battle.")
            return

        army_a, army_b = self.panel_a.army(), self.panel_b.army()
        if not army_a or not army_b:
            QMessageBox.warning(self, "Empty army", "Both armies need at least one unit.")
            return

        
        error = battles._winner_error( # Winner must have survivors, loser must have none
            winner,
            sum(self.panel_a.remaining().values()),
            sum(self.panel_b.remaining().values()),
        )
        if error:
            QMessageBox.warning(self, "Inconsistent result", error)
            return

        try:
            battle_id = battles.record(
                self.conn, winner,
                army_a=army_a, army_b=army_b,
                remaining_a=self.panel_a.remaining(),
                remaining_b=self.panel_b.remaining(),
                damage_dealt_a=self.panel_a.damage_by_unit(),
                damage_dealt_b=self.panel_b.damage_by_unit(),
                artifacts_a=self.panel_a.artifact_ids(),
                artifacts_b=self.panel_b.artifact_ids(),
                notes=self.notes_edit.text().strip() or None,
            )
        except (ValueError, sqlite3.Error) as e:
            QMessageBox.critical(self, "Couldn't save battle", str(e))
            return

        if self.carry_check.isChecked() and winner in ("A", "B"):
            self.carry_winner_forward(winner)
            self.status.setText(f"Saved battle #{battle_id}. Army {winner} carried over.")
        else:
            self.clear_form()
            self.status.setText(f"Saved battle #{battle_id}")
        self.saved.emit()