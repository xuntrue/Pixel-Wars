import sqlite3

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QAbstractItemView, QDoubleSpinBox, QFormLayout, QHBoxLayout, QHeaderView,
    QLineEdit, QMessageBox, QPushButton, QTableWidget, QTableWidgetItem,
    QVBoxLayout, QWidget,
)

from src.catalog import units

class UnitsTab(QWidget):
    COLUMNS = ["Name", "Damage", "Speed", "Health"]
    changed = Signal()

    def __init__(self, conn):
        super().__init__()
        self.conn = conn

        # --- input form ---
        self.name_edit = QLineEdit()
        self.damage_spin = self._make_spin(minimum=0.0)
        self.speed_spin = self._make_spin(minimum=0.0)
        self.health_spin = self._make_spin(minimum=0.01)

        form = QFormLayout()
        form.addRow("Name:", self.name_edit)
        form.addRow("Damage:", self.damage_spin)
        form.addRow("Speed:", self.speed_spin)
        form.addRow("Health:", self.health_spin)

        add_button = QPushButton("Add unit")
        add_button.clicked.connect(self.add_unit)

        delete_button = QPushButton("Delete selected")
        delete_button.clicked.connect(self.delete_unit)

        buttons = QHBoxLayout()
        buttons.addWidget(add_button)
        buttons.addWidget(delete_button)
        buttons.addStretch()

        # --- table ---
        self.table = QTableWidget(0, len(self.COLUMNS))
        self.table.setHorizontalHeaderLabels(self.COLUMNS)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)

        # --- assemble ---
        layout = QVBoxLayout(self)
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
        for r, unit in enumerate(rows):
            name_item = QTableWidgetItem(unit["name"])
            name_item.setData(Qt.UserRole, unit["id"])  # stash the DB id
            self.table.setItem(r, 0, name_item)
            self.table.setItem(r, 1, QTableWidgetItem(str(unit["damage"])))
            self.table.setItem(r, 2, QTableWidgetItem(str(unit["speed"])))
            self.table.setItem(r, 3, QTableWidgetItem(str(unit["health"])))

    def add_unit(self):
        name = self.name_edit.text().strip()
        if not name:
            QMessageBox.warning(self, "Missing name", "Enter a unit name.")
            return
        try:
            units.create(
                self.conn, name,
                self.damage_spin.value(),
                self.speed_spin.value(),
                self.health_spin.value(),
            )
        except sqlite3.IntegrityError:
            QMessageBox.warning(self, "Duplicate", f"A unit named '{name}' already exists.")
            return
        self.name_edit.clear()
        self.refresh()

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
        
        self.refresh()

        self.changed.emit()