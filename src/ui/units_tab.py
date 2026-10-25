import sqlite3

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QAbstractItemView, QComboBox, QDoubleSpinBox, QFormLayout, QHBoxLayout,
    QHeaderView, QLabel, QLineEdit, QMessageBox, QPushButton, QTableWidget,
    QTableWidgetItem, QVBoxLayout, QWidget,
)

from src.api.database import ATTACK_SPEEDS
from src.catalog import units


class UnitsTab(QWidget):
    COLUMNS = ["Name", "Damage", "Move speed", "Attack speed", "Health"]
    changed = Signal()

    def __init__(self, conn):
        super().__init__()
        self.conn = conn
        self.editing_id = None  # id of the unit being edited, or None when adding

        # --- banners ---
        self.mode_label = QLabel()
        self.mode_label.hide()
        self.missing_label = QLabel()
        self.missing_label.setStyleSheet("color: red;")
        self.missing_label.hide()

        # --- input form ---
        self.name_edit = QLineEdit()
        self.damage_spin = self._make_spin(minimum=0.0)
        self.speed_spin = self._make_spin(minimum=0.0)
        self.attack_speed_box = QComboBox()
        self.attack_speed_box.addItem("Select attack speed...")  # index 0 = nothing chosen
        self.attack_speed_box.addItems(list(ATTACK_SPEEDS))
        self.health_spin = self._make_spin(minimum=0.01)

        form = QFormLayout()
        form.addRow("Name:", self.name_edit)
        form.addRow("Damage:", self.damage_spin)
        form.addRow("Move speed:", self.speed_spin)
        form.addRow("Attack speed:", self.attack_speed_box)
        form.addRow("Health:", self.health_spin)

        # --- buttons ---
        self.submit_button = QPushButton("Add unit")
        self.submit_button.clicked.connect(self.submit)

        self.cancel_button = QPushButton("Cancel edit")
        self.cancel_button.clicked.connect(self.cancel_edit)
        self.cancel_button.hide()

        edit_button = QPushButton("Edit selected")
        edit_button.clicked.connect(self.edit_selected)

        delete_button = QPushButton("Delete selected")
        delete_button.clicked.connect(self.delete_unit)

        buttons = QHBoxLayout()
        buttons.addWidget(self.submit_button)
        buttons.addWidget(self.cancel_button)
        buttons.addWidget(edit_button)
        buttons.addWidget(delete_button)
        buttons.addStretch()

        # --- table ---
        self.table = QTableWidget(0, len(self.COLUMNS))
        self.table.setHorizontalHeaderLabels(self.COLUMNS)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.itemDoubleClicked.connect(lambda item: self.start_edit(item.row()))

        # --- assemble ---
        layout = QVBoxLayout(self)
        layout.addWidget(self.missing_label)
        layout.addWidget(self.mode_label)
        layout.addLayout(form)
        layout.addLayout(buttons)
        layout.addWidget(self.table)

        self.refresh()

    @staticmethod
    def _make_spin(minimum: float) -> QDoubleSpinBox:
        spin = QDoubleSpinBox()
        spin.setRange(minimum, 1_000_000)
        spin.setDecimals(2)
        return spin

    def refresh(self):
        """Reload the table from the database."""
        rows = units.list_all(self.conn)
        self.table.setRowCount(len(rows))
        missing = 0
        for r, unit in enumerate(rows):
            name_item = QTableWidgetItem(unit["name"])
            name_item.setData(Qt.UserRole, unit["id"])  # stash the DB id
            self.table.setItem(r, 0, name_item)
            self.table.setItem(r, 1, QTableWidgetItem(str(unit["damage"])))
            self.table.setItem(r, 2, QTableWidgetItem(str(unit["speed"])))

            if unit["attack_speed"] is None:
                missing += 1
                attack_item = QTableWidgetItem("(not set)")
                attack_item.setForeground(QColor("red"))
            else:
                attack_item = QTableWidgetItem(unit["attack_speed"])
            self.table.setItem(r, 3, attack_item)
            self.table.setItem(r, 4, QTableWidgetItem(str(unit["health"])))

        if missing:
            self.missing_label.setText(
                f"{missing} unit(s) still need an attack speed. Double-click a row to set it."
            )
            self.missing_label.show()
        else:
            self.missing_label.hide()

    # ---------- add / edit ----------
    def submit(self):
        """The main button: adds a new unit, or saves changes when editing."""
        name = self.name_edit.text().strip()
        if not name:
            QMessageBox.warning(self, "Missing name", "Enter a unit name.")
            return
        index = self.attack_speed_box.currentIndex()
        if index == 0:
            QMessageBox.warning(self, "Missing attack speed", "Select an attack speed.")
            return
        attack_speed = ATTACK_SPEEDS[index - 1]
        damage = self.damage_spin.value()
        speed = self.speed_spin.value()
        health = self.health_spin.value()

        try:
            if self.editing_id is None:
                units.create(self.conn, name, damage, speed, health, attack_speed)
            else:
                if not self._confirm_stat_change(damage, speed, health, attack_speed):
                    return
                units.update(self.conn, self.editing_id, name, damage, speed, health,
                             attack_speed)
        except sqlite3.IntegrityError:
            QMessageBox.warning(self, "Duplicate", f"A unit named '{name}' already exists.")
            return

        if self.editing_id is None:
            self.name_edit.clear()  # keep the stats: handy when adding similar units
        else:
            self._leave_edit_mode()
        self.refresh()
        self.changed.emit()

    def _confirm_stat_change(self, damage, speed, health, attack_speed) -> bool:
        """Ask before changing the stats of a unit that past battles used."""
        old = units.get(self.conn, self.editing_id)
        if old is None:  # deleted elsewhere in the meantime
            QMessageBox.warning(self, "Unit not found", "This unit no longer exists.")
            self._leave_edit_mode()
            self.refresh()
            return False
        old_numbers = tuple(round(v, 2) for v in (old["damage"], old["speed"], old["health"]))
        new_numbers = tuple(round(v, 2) for v in (damage, speed, health))

        # Filling in a blank attack speed is not a change to history; altering one is
        attack_changed = (old["attack_speed"] is not None
                          and old["attack_speed"] != attack_speed)
        used = units.battle_count(self.conn, self.editing_id)
        if used == 0 or (old_numbers == new_numbers and not attack_changed):
            return True
        answer = QMessageBox.question(
            self, "Change unit stats",
            f"'{old['name']}' appears in {used} recorded battle(s).\n\n"
            "Changing its damage, speed, attack speed or health changes how those "
            "battles are analysed. Continue?",
        )
        return answer == QMessageBox.Yes

    def edit_selected(self):
        row = self.table.currentRow()
        if row < 0:
            QMessageBox.information(self, "Nothing selected", "Select a unit first.")
            return
        self.start_edit(row)

    def start_edit(self, row: int):
        unit_id = self.table.item(row, 0).data(Qt.UserRole)
        unit = units.get(self.conn, unit_id)
        if unit is None:
            self.refresh()
            return
        self.editing_id = unit_id
        self.name_edit.setText(unit["name"])
        self.damage_spin.setValue(unit["damage"])
        self.speed_spin.setValue(unit["speed"])
        self.health_spin.setValue(unit["health"])
        value = unit["attack_speed"]
        self.attack_speed_box.setCurrentIndex(0 if value is None else ATTACK_SPEEDS.index(value) + 1)
        self.submit_button.setText("Save changes")
        self.cancel_button.show()
        self.mode_label.setText(f"<b>Editing '{unit['name']}'</b>")
        self.mode_label.show()
        self.name_edit.setFocus()

    def cancel_edit(self):
        self._leave_edit_mode()

    def _leave_edit_mode(self):
        self.editing_id = None
        self.name_edit.clear()
        for spin in (self.damage_spin, self.speed_spin, self.health_spin):
            spin.setValue(spin.minimum())
        self.attack_speed_box.setCurrentIndex(0)
        self.submit_button.setText("Add unit")
        self.cancel_button.hide()
        self.mode_label.hide()

    # ---------- delete ----------
    def delete_unit(self):
        row = self.table.currentRow()
        if row < 0:
            QMessageBox.information(self, "Nothing selected", "Select a unit first.")
            return
        item = self.table.item(row, 0)
        unit_id = item.data(Qt.UserRole)

        answer = QMessageBox.question(self, "Delete unit", f"Delete '{item.text()}'?")
        if answer != QMessageBox.Yes:
            return
        try:
            units.delete(self.conn, unit_id)
        except sqlite3.IntegrityError:
            QMessageBox.warning(
                self, "Can't delete",
                "This unit is used in recorded battles, so it can't be deleted.",
            )
            return
        if unit_id == self.editing_id:
            self._leave_edit_mode()  # the unit being edited no longer exists
        self.refresh()
        self.changed.emit()