from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QAbstractItemView, QHBoxLayout, QHeaderView, QLabel, QLineEdit, QListWidget,
    QListWidgetItem, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget,
)

from src.catalog import abilities, unit_abilities, units


class UnitAbilitiesTab(QWidget):
    changed = Signal()  # emitted after a unit's abilities are changed

    def __init__(self, conn):
        super().__init__()
        self.conn = conn
        self.current_unit_id = None
        self.all_abilities = []

        # --- left: units and a summary of their abilities ---
        self.filter_edit = QLineEdit()
        self.filter_edit.setPlaceholderText("Filter units...")
        self.filter_edit.textChanged.connect(self.apply_filter)

        self.unit_table = QTableWidget(0, 2)
        self.unit_table.setHorizontalHeaderLabels(["Unit", "Abilities"])
        self.unit_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.unit_table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.unit_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        header = self.unit_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.Stretch)
        self.unit_table.itemSelectionChanged.connect(self.on_unit_selected)

        left = QVBoxLayout()
        left.addWidget(self.filter_edit)
        left.addWidget(self.unit_table)

        # --- right: the checklist for the selected unit ---
        self.heading = QLabel("Select a unit")
        self.hint = QLabel("No abilities exist yet. Create some in the Abilities tab.")
        self.ability_list = QListWidget()
        self.ability_list.itemChanged.connect(self.on_ability_toggled)

        right = QVBoxLayout()
        right.addWidget(self.heading)
        right.addWidget(self.hint)
        right.addWidget(self.ability_list)

        layout = QHBoxLayout(self)
        layout.addLayout(left, stretch=3)
        layout.addLayout(right, stretch=2)

        self.reload()

    def reload(self):
        """Reload everything from the database. Keeps the selected unit if it still exists."""
        selected = self.current_unit_id
        self.all_abilities = abilities.list_all(self.conn)
        mapping = unit_abilities.names_by_unit(self.conn)
        rows = units.list_all(self.conn)

        self.unit_table.blockSignals(True)
        self.unit_table.setRowCount(len(rows))
        found = None
        for r, unit in enumerate(rows):
            name_item = QTableWidgetItem(unit["name"])
            name_item.setData(Qt.UserRole, unit["id"])
            self.unit_table.setItem(r, 0, name_item)
            self.unit_table.setItem(
                r, 1, QTableWidgetItem(", ".join(mapping.get(unit["id"], [])))
            )
            if unit["id"] == selected:
                found = r
        if found is not None:
            self.unit_table.selectRow(found)
        else:
            self.unit_table.clearSelection()
        self.unit_table.blockSignals(False)

        self.apply_filter(self.filter_edit.text())
        self.show_unit(selected if found is not None else None)

    def apply_filter(self, text: str):
        text = text.strip().lower()
        for row in range(self.unit_table.rowCount()):
            name = self.unit_table.item(row, 0).text().lower()
            self.unit_table.setRowHidden(row, text not in name)

    def on_unit_selected(self):
        selected_rows = self.unit_table.selectionModel().selectedRows()
        unit_id = None
        if selected_rows:
            unit_id = self.unit_table.item(selected_rows[0].row(), 0).data(Qt.UserRole)
        self.show_unit(unit_id)

    def show_unit(self, unit_id):
        """Fill the checklist with all abilities, ticking the unit's own."""
        self.current_unit_id = unit_id
        self.ability_list.blockSignals(True)  # don't save while we're just displaying
        self.ability_list.clear()

        if unit_id is None:
            self.heading.setText("Select a unit")
        else:
            unit = units.get(self.conn, unit_id)
            self.heading.setText(f"<b>{unit['name']}</b>: tick its abilities")
            owned = unit_abilities.ability_ids_for_unit(self.conn, unit_id)
            for ability in self.all_abilities:
                item = QListWidgetItem(ability["name"])
                item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
                item.setCheckState(Qt.Checked if ability["id"] in owned else Qt.Unchecked)
                item.setData(Qt.UserRole, ability["id"])
                self.ability_list.addItem(item)

        self.ability_list.blockSignals(False)
        self.ability_list.setEnabled(unit_id is not None)
        self.hint.setVisible(not self.all_abilities)

    def on_ability_toggled(self, _item):
        """Save the unit's abilities every time a box is ticked or unticked."""
        if self.current_unit_id is None:
            return
        checked = [self.ability_list.item(i)
                   for i in range(self.ability_list.count())
                   if self.ability_list.item(i).checkState() == Qt.Checked]
        unit_abilities.set_for_unit(
            self.conn, self.current_unit_id, [item.data(Qt.UserRole) for item in checked]
        )
        row = self.unit_table.currentRow()
        if row >= 0:
            self.unit_table.item(row, 1).setText(", ".join(item.text() for item in checked))
        self.changed.emit()