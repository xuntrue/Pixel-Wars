import sys
import sqlite3

from datetime import datetime
from PySide6.QtWidgets import QApplication

from src.analysis.analyse import analyse
from src.api.database import DATA_DIR, get_connection, init_db
from src.catalog import units, artifacts
from src.battles import battles
from src.ui.main_window import MainWindow

def run_application():
    conn = get_connection()
    init_db(conn)

    app = QApplication(sys.argv)
    window = MainWindow(conn)
    window.show()
    sys.exit(
        app.exec() # starts the event loop
    )

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

if __name__ == "__main__":
    clear_all_battles()