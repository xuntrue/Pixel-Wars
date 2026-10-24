import sys
from PySide6.QtWidgets import QApplication

from src.api.database import get_connection, init_db
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

if __name__ == "__main__":
    run_application()