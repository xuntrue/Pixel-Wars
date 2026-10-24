import sqlite3

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QHBoxLayout, QInputDialog, QLineEdit, QListWidget, QListWidgetItem,
    QMessageBox, QPushButton, QVBoxLayout, QWidget,
)

from src.catalog import artifacts

class ArtifactsTab(QWidget):
    changed = Signal()  # emitted after an artifact is added, renamed, or deleted

    def __init__(self, conn):
        super().__init__()
        self.conn = conn

        # --- input row ---
        self.name_edit = QLineEdit()
        self.name_edit.setPlaceholderText("New artifact name")
        self.name_edit.returnPressed.connect(self.add_artifact)  # Enter key works too

        add_button = QPushButton("Add artifact")
        add_button.clicked.connect(self.add_artifact)

        input_row = QHBoxLayout()
        input_row.addWidget(self.name_edit)
        input_row.addWidget(add_button)

        # --- list ---
        self.list_widget = QListWidget()
        self.list_widget.itemDoubleClicked.connect(self.rename_artifact)

        delete_button = QPushButton("Delete selected")
        delete_button.clicked.connect(self.delete_artifact)

        # --- assemble ---
        layout = QVBoxLayout(self)
        layout.addLayout(input_row)
        layout.addWidget(self.list_widget)
        layout.addWidget(delete_button)

        self.refresh()

    def refresh(self):
        self.list_widget.clear()
        for artifact in artifacts.list_all(self.conn):
            item = QListWidgetItem(artifact["name"])
            item.setData(Qt.UserRole, artifact["id"])
            self.list_widget.addItem(item)

    def add_artifact(self):
        name = self.name_edit.text().strip()
        if not name:
            return
        try:
            artifacts.create(self.conn, name)
        except sqlite3.IntegrityError:
            QMessageBox.warning(self, "Duplicate", f"An artifact named '{name}' already exists.")
            return
        self.name_edit.clear()
        self.refresh()
        self.changed.emit()

    def rename_artifact(self, item: QListWidgetItem):
        """Triggered by double-clicking an item."""
        new_name, ok = QInputDialog.getText(
            self, "Rename artifact", "New name:", text=item.text()
        )
        new_name = new_name.strip()
        if not ok or not new_name or new_name == item.text():
            return
        try:
            artifacts.rename(self.conn, item.data(Qt.UserRole), new_name)
        except sqlite3.IntegrityError:
            QMessageBox.warning(self, "Duplicate", f"An artifact named '{new_name}' already exists.")
            return
        self.refresh()
        self.changed.emit()

    def delete_artifact(self):
        item = self.list_widget.currentItem()
        if item is None:
            QMessageBox.information(self, "Nothing selected", "Select an artifact first.")
            return
        answer = QMessageBox.question(self, "Delete artifact", f"Delete '{item.text()}'?")
        if answer != QMessageBox.Yes:
            return
        try:
            artifacts.delete(self.conn, item.data(Qt.UserRole))
        except sqlite3.IntegrityError:
            QMessageBox.warning(
                self, "Can't delete",
                "This artifact is used in recorded battles, so it can't be deleted.",
            )
            return
        self.refresh()
        self.changed.emit()