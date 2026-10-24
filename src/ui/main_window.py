from PySide6.QtWidgets import QMainWindow, QTabWidget

from .artifacts_tab import ArtifactsTab
from .record_battle_tab import RecordBattleTab
from .units_tab import UnitsTab


class MainWindow(QMainWindow):
    def __init__(self, conn):
        super().__init__()
        self.setWindowTitle("Pixel Wars Tracker")

        self.resize(1000, 650)

        self.units_tab = UnitsTab(conn)
        self.artifacts_tab = ArtifactsTab(conn)
        self.record_tab = RecordBattleTab(conn)
        
        # If or when the catalog changes, battle form rebuilds itself
        self.units_tab.changed.connect(self.record_tab.rebuild)
        self.artifacts_tab.changed.connect(self.record_tab.rebuild)

        tabs = QTabWidget()
        tabs.addTab(self.record_tab, "Record Battle")
        tabs.addTab(self.units_tab, "Units")
        tabs.addTab(self.artifacts_tab, "Artifacts")

        self.setCentralWidget(tabs)