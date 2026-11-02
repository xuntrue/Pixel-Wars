from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtWidgets import (
    QComboBox, QCompleter, QDoubleSpinBox, QGroupBox, QHBoxLayout, QLabel,
    QPushButton, QScrollArea, QSpinBox, QVBoxLayout, QWidget,
)

from src.catalog import artifacts, units

SPIN_WIDTH = 90
DAMAGE_WIDTH = 120
DELETE_WIDTH = 26

class ChoiceCombo(QComboBox):
    """Searchable dropdown of catalog items (each choice has an 'id' and a 'name')."""

    def __init__(self):
        super().__init__()
        self.setEditable(True)
        self.setInsertPolicy(QComboBox.NoInsert)  # typing filters; it never adds items
        self.setMinimumWidth(130)
        completer = self.completer()
        completer.setCompletionMode(QCompleter.PopupCompletion)
        completer.setFilterMode(Qt.MatchContains)
        completer.setCaseSensitivity(Qt.CaseInsensitive)
        self.lineEdit().editingFinished.connect(self._commit_text)

    def selected_id(self):
        return self.currentData()

    def set_choices(self, choices, selected_id=None):
        """Replace the list of choices, keeping the current selection."""
        if selected_id is None:
            selected_id = self.selected_id()
        self.blockSignals(True)
        self.clear()
        for choice in choices:
            self.addItem(choice["name"], choice["id"])
        self.setCurrentIndex(self.findData(selected_id))
        self.blockSignals(False)

    def _commit_text(self):
        """When typing ends: accept an exact name, otherwise revert the text."""
        index = self.findText(self.currentText(), Qt.MatchFixedString)
        if index >= 0:
            self.setCurrentIndex(index)
        else:
            self.setEditText(self.itemText(self.currentIndex()))


def _delete_button(tooltip: str) -> QPushButton:
    button = QPushButton("✕")
    button.setFixedWidth(DELETE_WIDTH)
    button.setToolTip(tooltip)
    # Keep the column's width when hidden so rows stay aligned.
    policy = button.sizePolicy()
    policy.setRetainSizeWhenHidden(True)
    button.setSizePolicy(policy)
    return button


class UnitRow(QWidget):
    """One unit record: type, deployed, survivors, damage dealt."""

    type_changed = Signal()
    values_changed = Signal()
    delete_requested = Signal(object)

    def __init__(self):
        super().__init__()
        self.combo = ChoiceCombo()

        self.deployed = QSpinBox()
        self.deployed.setRange(1, 999)
        self.deployed.setFixedWidth(SPIN_WIDTH)

        self.survivors = QSpinBox()
        self.survivors.setRange(0, self.deployed.value())
        self.survivors.setFixedWidth(SPIN_WIDTH)

        self.damage = QDoubleSpinBox()
        self.damage.setRange(0, 10_000_000)
        self.damage.setDecimals(2)
        self.damage.setFixedWidth(DAMAGE_WIDTH)

        self.delete_button = _delete_button("Remove this unit")

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)
        layout.addWidget(self.combo, stretch=1)
        layout.addWidget(self.deployed)
        layout.addWidget(self.survivors)
        layout.addWidget(self.damage)
        layout.addWidget(self.delete_button)

        self.combo.currentIndexChanged.connect(lambda *_: self.type_changed.emit())
        self.deployed.valueChanged.connect(self.survivors.setMaximum)  # survivors <= deployed
        for spin in (self.deployed, self.survivors, self.damage):
            spin.valueChanged.connect(lambda *_: self.values_changed.emit())
        self.delete_button.clicked.connect(lambda: self.delete_requested.emit(self))

    def set_deletable(self, deletable: bool):
        self.delete_button.setVisible(deletable)

class ArtifactRow(QWidget):
    """One artifact record: just a type and a delete button."""

    type_changed = Signal()
    delete_requested = Signal(object)

    def __init__(self):
        super().__init__()
        self.combo = ChoiceCombo()
        self.delete_button = _delete_button("Remove this artifact")

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)
        layout.addWidget(self.combo, stretch=1)
        layout.addWidget(self.delete_button)

        self.combo.currentIndexChanged.connect(lambda *_: self.type_changed.emit())
        self.delete_button.clicked.connect(lambda: self.delete_requested.emit(self))


class ArmyPanel(QGroupBox):
    """One side of a battle, built from dynamic unit and artifact records."""

    def __init__(self, title: str, conn):
        super().__init__(title)
        self.conn = conn
        self.all_units = []
        self.all_artifacts = []
        self.unit_rows: list[UnitRow] = []
        self.artifact_rows: list[ArtifactRow] = []

        # --- units section ---
        self.header = self._make_header()
        self.no_units_label = QLabel("No units yet. Add some in the Units tab.")
        self.units_layout = QVBoxLayout()
        self.units_layout.setSpacing(4)
        self.add_unit_button = QPushButton("Add Unit")
        self.add_unit_button.clicked.connect(lambda: self._add_unit_row())

        # --- artifacts section ---
        self.no_artifacts_label = QLabel("No artifacts yet. Add some in the Artifacts tab.")
        self.artifacts_layout = QVBoxLayout()
        self.artifacts_layout.setSpacing(4)
        self.add_artifact_button = QPushButton("Add Artifact")
        self.add_artifact_button.clicked.connect(lambda: self._add_artifact_row())

        content = QWidget()
        layout = QVBoxLayout(content)
        layout.addWidget(self.header)
        layout.addWidget(self.no_units_label)
        layout.addLayout(self.units_layout)
        layout.addWidget(self.add_unit_button, alignment=Qt.AlignLeft)
        layout.addSpacing(12)
        layout.addWidget(QLabel("<b>Artifacts</b>"))
        layout.addWidget(self.no_artifacts_label)
        layout.addLayout(self.artifacts_layout)
        layout.addWidget(self.add_artifact_button, alignment=Qt.AlignLeft)
        layout.addStretch()

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(content)

        self.total_label = QLabel()

        outer = QVBoxLayout(self)
        outer.addWidget(scroll)
        outer.addWidget(self.total_label)

        self.rebuild()

    @staticmethod
    def _make_header() -> QWidget:
        header = QWidget()
        layout = QHBoxLayout(header)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)
        layout.addWidget(QLabel("<b>Unit type</b>"), stretch=1)
        for text, width in (("Deployed", SPIN_WIDTH), ("Survivors", SPIN_WIDTH),
                            ("Damage dealt", DAMAGE_WIDTH)):
            label = QLabel(f"<b>{text}</b>")
            label.setFixedWidth(width)
            layout.addWidget(label)
        spacer = QWidget()
        spacer.setFixedWidth(DELETE_WIDTH)
        layout.addWidget(spacer)
        return header

    # ---------- catalog changes ----------
    def rebuild(self):
        """Reload the catalog. Records already filled in are kept."""
        self.all_units = units.list_all(self.conn)
        self.all_artifacts = artifacts.list_all(self.conn)

        # Drop records whose unit/artifact was deleted from the catalog.
        unit_ids = {u["id"] for u in self.all_units}
        for row in list(self.unit_rows):
            if row.combo.selected_id() not in unit_ids:
                self._remove_unit_row(row)
        artifact_ids = {a["id"] for a in self.all_artifacts}
        for row in list(self.artifact_rows):
            if row.combo.selected_id() not in artifact_ids:
                self._remove_artifact_row(row)

        if not self.unit_rows and self.all_units:
            self._add_unit_row()  # an army always has at least one unit record
        self._sync()

    # ---------- adding / removing records ----------
    def _add_unit_row(self):
        used = {r.combo.selected_id() for r in self.unit_rows}
        unit_id = next((u["id"] for u in self.all_units if u["id"] not in used), None)
        if unit_id is None:
            return
        row = UnitRow()
        row.combo.set_choices(self.all_units, unit_id)
        row.type_changed.connect(self._schedule_refresh)
        row.values_changed.connect(self._update_total)
        row.delete_requested.connect(self._remove_unit_row)
        self.unit_rows.append(row)
        self.units_layout.addWidget(row)
        self._sync()

    def _remove_unit_row(self, row: UnitRow):
        self.unit_rows.remove(row)
        self.units_layout.removeWidget(row)
        row.hide()
        row.deleteLater()
        self._sync()

    def _add_artifact_row(self):
        used = {r.combo.selected_id() for r in self.artifact_rows}
        artifact_id = next((a["id"] for a in self.all_artifacts if a["id"] not in used), None)
        if artifact_id is None:
            return
        row = ArtifactRow()
        row.combo.set_choices(self.all_artifacts, artifact_id)
        row.type_changed.connect(self._schedule_refresh)
        row.delete_requested.connect(self._remove_artifact_row)
        self.artifact_rows.append(row)
        self.artifacts_layout.addWidget(row)
        self._sync()

    def _remove_artifact_row(self, row: ArtifactRow):
        self.artifact_rows.remove(row)
        self.artifacts_layout.removeWidget(row)
        row.hide()
        row.deleteLater()
        self._sync()

   # ---------- keeping the UI consistent ----------
    def _schedule_refresh(self):
        # Deferred so a dropdown isn't rebuilt while it is still handling its own click.
        QTimer.singleShot(0, self._refresh_choices)

    def _refresh_choices(self):
        """Each dropdown offers its own selection plus types no other record uses."""
        for rows, catalog in ((self.unit_rows, self.all_units),
                              (self.artifact_rows, self.all_artifacts)):
            used = {r.combo.selected_id() for r in rows}
            for row in rows:
                mine = row.combo.selected_id()
                row.combo.set_choices(
                    [c for c in catalog if c["id"] == mine or c["id"] not in used], mine
                )

    def _sync(self):
        """Refresh everything that depends on which records exist."""
        self._refresh_choices()
        for index, row in enumerate(self.unit_rows):
            row.set_deletable(index > 0)  # the first unit record can't be removed

        self.header.setVisible(bool(self.all_units))
        self.no_units_label.setVisible(not self.all_units)
        self.add_unit_button.setVisible(0 < len(self.unit_rows) < len(self.all_units))

        self.no_artifacts_label.setVisible(not self.all_artifacts)
        self.add_artifact_button.setVisible(len(self.artifact_rows) < len(self.all_artifacts))
        self._update_total()

    def _update_total(self):
        deployed = sum(r.deployed.value() for r in self.unit_rows)
        survivors = sum(r.survivors.value() for r in self.unit_rows)
        damage = sum(r.damage.value() for r in self.unit_rows)
        self.total_label.setText(
            f"Totals: {deployed} deployed, {survivors} surviving, {damage:,.2f} damage dealt"
        )

    # ---------- reading values (used by the Record Battle tab) ----------
    def army(self) -> dict[int, int]:
        return {r.combo.selected_id(): r.deployed.value() for r in self.unit_rows}

    def remaining(self) -> dict[int, int]:
        return {r.combo.selected_id(): r.survivors.value()
                for r in self.unit_rows if r.survivors.value() > 0}

    def damage_by_unit(self) -> dict[int, float]:
        return {r.combo.selected_id(): r.damage.value() for r in self.unit_rows}

    def artifact_ids(self) -> list[int]:
        return [r.combo.selected_id() for r in self.artifact_rows]

    def keep_survivors(self):
        """ Keep only surviving unit types, with the survivor count as the new deployed count, plus the artifacts """
        keep = [row for row in self.unit_rows if row.survivors.value() > 0]
        if not keep:
            self.clear()
            return
        for row in list(self.unit_rows):
            if row not in keep:
                self._remove_unit_row(row)
        for row in keep:
            survivors = row.survivors.value()
            row.survivors.setValue(0)
            row.damage.setValue(0)
            row.deployed.setValue(survivors)

    def clear(self):
        """ Back to the starting state: one unit record, no artifacts """
        for row in list(self.unit_rows):
            self._remove_unit_row(row)
        for row in list(self.artifact_rows):
            self._remove_artifact_row(row)
        if self.all_units:
            self._add_unit_row()