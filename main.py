import sys
import sqlite3

from datetime import datetime
from PySide6.QtWidgets import QApplication

from src.analysis.analyse import analyse
from src.api.database import DATA_DIR, get_connection, init_db
from src.catalog import units, artifacts
from src.battles import battles

from src.ui.catalog_window import CatalogWindow
from src.ui.record_window import RecordWindow

def _run_window(window_class) -> None:
    conn = get_connection()
    init_db(conn)

    app = QApplication(sys.argv[:1]) 
    # DO NOT pass our own command-line word on to Qt
    
    window = window_class(conn)
    window.show()
    exit_code = app.exec()  # Starts the event loop
    conn.close()
    sys.exit(exit_code)

def run_recorder() -> None:
    _run_window(RecordWindow)


def run_catalog_editor() -> None:
    _run_window(CatalogWindow)

def clear_all_battles() -> None:
    """ Delete every recorded battle (units and artifacts are kept). Backs up first. """
    conn = get_connection()
    init_db(conn)
    try:
        total = battles.count(conn)
        if total == 0:
            print("There are no battles to delete.")
            return

        answer = input(f"This permanently deletes all {total} battles. "
                       "Type DELETE to continue: ")
        if answer.strip() != "DELETE":
            print("Cancelled. Nothing was deleted.")
            return

        backup_path = DATA_DIR / "backups" / f"pixel_wars_{datetime.now():%Y%m%d_%H%M%S}.db"
        backup_path.parent.mkdir(parents=True, exist_ok=True)
        backup = sqlite3.connect(backup_path)
        try:
            conn.backup(backup)
        finally:
            backup.close()
        print(f"Backup saved to {backup_path}")

        deleted = battles.delete_all(conn)
        print(f"Deleted {deleted} battles. Units and artifacts were kept.")
    finally:
        conn.close()

COMMANDS = {
    "record": run_recorder,
    "catalog": run_catalog_editor,
    "analyse": analyse,
    #"train": train,
    "clear-battles": clear_all_battles,
}

if __name__ == "__main__":
    #run_recorder()
    run_catalog_editor()
    #analyse()

    """
    command = sys.argv[1] if len(sys.argv) > 1 else "record"
    if command not in COMMANDS:
        print(f"Unknown command '{command}'. Choose one of: {', '.join(COMMANDS)}")
        sys.exit(1)
    COMMANDS[command]()
    """