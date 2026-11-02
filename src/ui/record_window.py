from PySide6.QtGui import QAction, QKeySequence
from PySide6.QtWidgets import QMainWindow, QTabWidget

from .baseline_tab import BaselineTab
from .record_battle_tab import RecordBattleTab

class RecordWindow(QMainWindow):
    """ Record battles. Units and artifacts are only read from the database."""

    def __init__(self, conn):
        super().__init__()
        self.setWindowTitle("Pixel Wars: Record Battles")
        self.resize(1000, 650)

        self.record_tab = RecordBattleTab(conn)
        self.baseline_tab = BaselineTab(conn)
        
        tabs = QTabWidget()
        tabs.addTab(self.record_tab, "Record Battle")
        tabs.addTab(self.baseline_tab, "1v1 Baseline")
        self.setCentralWidget(tabs)

        # Optional: pick up units/artifacts changed in the catalog app without restarting.
        reload_action = QAction("Reload catalog (F5)", self)
        reload_action.setShortcut(QKeySequence("F5"))
        reload_action.triggered.connect(self.reload_catalog)
        self.menuBar().addAction(reload_action)

    def reload_catalog(self):
        self.record_tab.rebuild()
        self.baseline_tab.reload()