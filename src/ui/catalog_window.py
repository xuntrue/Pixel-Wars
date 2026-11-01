from PySide6.QtWidgets import QMainWindow, QTabWidget

from .artifacts_tab import ArtifactsTab
from .units_tab import UnitsTab


class CatalogWindow(QMainWindow):
    """ Define units and artifacts """

    def __init__(self, conn):
        super().__init__()
        self.setWindowTitle("Pixel Wars: Units & Artifacts")
        self.resize(900, 600)

        tabs = QTabWidget()
        tabs.addTab(UnitsTab(conn), "Units")
        tabs.addTab(ArtifactsTab(conn), "Artifacts")
        self.setCentralWidget(tabs)