import sqlite3

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QHBoxLayout, QInputDialog, QLineEdit, QListWidget, QListWidgetItem,
    QMessageBox, QPushButton, QVBoxLayout, QWidget,
)

from src.catalog import abilities

class AbilitiesTab(QWidget):
    changed = Signal()  # emitted after an ability is added, renamed, or deleted

    def __init__(self, conn):
        super().__init__()
        self.conn = conn

        self.name_edit = QLineEdit()
        self.name_edit.setPlaceholderText("New ability name")
        self.name_edit.returnPressed.connect(self.add_ability)

        add_button = QPushButton("Add ability")
        add_button.clicked.connect(self.add_ability)

        input_row = QHBoxLayout()
        input_row.addWidget(self.name_edit)
        input_row.addWidget(add_button)

        self.list_widget = QListWidget()
        self.list_widget.itemDoubleClicked.connect(self.rename_ability)

        delete_button = QPushButton("Delete selected")
        delete_button.clicked.connect(self.delete_ability)

        layout = QVBoxLayout(self)
        layout.addLayout(input_row)
        layout.addWidget(self.list_widget)
        layout.addWidget(delete_button)

        self.refresh()

    def refresh(self):
        self.list_widget.clear()
        for ability in abilities.list_all(self.conn):
            item = QListWidgetItem(ability["name"])
            item.setData(Qt.UserRole, ability["id"])
            self.list_widget.addItem(item)

    def add_ability(self):
        name = self.name_edit.text().strip()
        if not name:
            return
        try:
            abilities.create(self.conn, name)
        except sqlite3.IntegrityError:
            QMessageBox.warning(self, "Duplicate", f"An ability named '{name}' already exists.")
            return
        self.name_edit.clear()
        self.refresh()
        self.changed.emit()

    def rename_ability(self, item: QListWidgetItem):
        """ Triggered by double-clicking an item """
        new_name, ok = QInputDialog.getText(
            self, "Rename ability", "New name:", text=item.text()
        )
        new_name = new_name.strip()
        if not ok or not new_name or new_name == item.text():
            return
        try:
            abilities.rename(self.conn, item.data(Qt.UserRole), new_name)
        except sqlite3.IntegrityError:
            QMessageBox.warning(self, "Duplicate", f"An ability named '{new_name}' already exists.")
            return
        self.refresh()
        self.changed.emit()

    def delete_ability(self):
        item = self.list_widget.currentItem()
        if item is None:
            QMessageBox.information(self, "Nothing selected", "Select an ability first.")
            return
        ability_id = item.data(Qt.UserRole)
        count = abilities.unit_count(self.conn, ability_id)
        extra = f"\n\nIt will be removed from {count} unit(s)." if count else ""
        answer = QMessageBox.question(self, "Delete ability", f"Delete '{item.text()}'?{extra}")
        if answer != QMessageBox.Yes:
            return
        abilities.delete(self.conn, ability_id)
        self.refresh()
        self.changed.emit()