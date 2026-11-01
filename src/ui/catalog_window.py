from PySide6.QtWidgets import QMainWindow, QTabWidget

from .abilities_tab import AbilitiesTab
from .artifacts_tab import ArtifactsTab
from .unit_abilities_tab import UnitAbilitiesTab
from .units_tab import UnitsTab

class CatalogWindow(QMainWindow):
    """ Define units, artifacts, and abilities """

    def __init__(self, conn):
        super().__init__()
        self.setWindowTitle("Pixel Wars: Units & Artifacts")
        self.resize(900, 600)

        self.units_tab = UnitsTab(conn)
        self.artifacts_tab = ArtifactsTab(conn)
        self.abilities_tab = AbilitiesTab(conn)
        self.unit_abilities_tab = UnitAbilitiesTab(conn)

        # The mapping tab lists units and abilities, so it reloads when either changes
        self.units_tab.changed.connect(self.unit_abilities_tab.reload)
        self.abilities_tab.changed.connect(self.unit_abilities_tab.reload)

        tabs = QTabWidget()
        tabs.addTab(self.units_tab, "Units")
        tabs.addTab(self.artifacts_tab, "Artifacts")
        tabs.addTab(self.abilities_tab, "Abilities")
        tabs.addTab(self.unit_abilities_tab, "Unit Abilities")

        self.setCentralWidget(tabs)