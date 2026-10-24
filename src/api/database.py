import sqlite3

from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
DB_PATH = DATA_DIR / "pixel_wars.db"

# Columns added after the first version of the schema.
MIGRATIONS = [
    ("battle_units", "remaining", "INTEGER CHECK (remaining >= 0)"),
    ("battle_units", "damage_dealt", "REAL CHECK (damage_dealt >= 0)"),
]

# Columns that no longer belong in the schema
OBSOLETE_COLUMNS = [
    ("battles", "damage_a"),
    ("battles", "damage_b"),
]

SCHEMA = """
CREATE TABLE IF NOT EXISTS units (
    id      INTEGER PRIMARY KEY,
    name    TEXT NOT NULL UNIQUE,
    damage  REAL NOT NULL CHECK (damage >= 0),
    speed   REAL NOT NULL CHECK (speed >= 0),
    health  REAL NOT NULL CHECK (health > 0)
);

CREATE TABLE IF NOT EXISTS artifacts (
    id    INTEGER PRIMARY KEY,
    name  TEXT NOT NULL UNIQUE
);

CREATE TABLE IF NOT EXISTS battles (
    id          INTEGER PRIMARY KEY,
    timestamp   TEXT NOT NULL DEFAULT (datetime('now')),
    winner      TEXT NOT NULL CHECK (winner IN ('A', 'B', 'draw')),
    carry_over  INTEGER NOT NULL DEFAULT 0 CHECK (carry_over IN (0, 1)),
    damage_a    REAL CHECK (damage_a >= 0),
    damage_b    REAL CHECK (damage_b >= 0),
    notes       TEXT
);

CREATE TABLE IF NOT EXISTS battle_units (
    battle_id     INTEGER NOT NULL REFERENCES battles(id) ON DELETE CASCADE,
    side          TEXT NOT NULL CHECK (side IN ('A', 'B')),
    unit_id       INTEGER NOT NULL REFERENCES units(id) ON DELETE RESTRICT,
    count         INTEGER NOT NULL CHECK (count > 0),
    remaining     INTEGER CHECK (remaining >= 0),
    damage_dealt  REAL CHECK (damage_dealt >= 0),
    PRIMARY KEY (battle_id, side, unit_id)
);

CREATE TABLE IF NOT EXISTS battle_artifacts (
    battle_id    INTEGER NOT NULL REFERENCES battles(id) ON DELETE CASCADE,
    side         TEXT NOT NULL CHECK (side IN ('A', 'B')),
    artifact_id  INTEGER NOT NULL REFERENCES artifacts(id) ON DELETE RESTRICT,
    PRIMARY KEY (battle_id, side, artifact_id)
);
"""

def get_connection(db_path: Path = DB_PATH) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")  # off by default in SQLite
    return conn

def _column_exists(conn: sqlite3.Connection, table: str, column: str) -> bool:
    return any(row["name"] == column
               for row in conn.execute(f"PRAGMA table_info({table})"))

def init_db(conn: sqlite3.Connection) -> None:
    conn.executescript(SCHEMA)
    for table, column, definition in MIGRATIONS:
        if not _column_exists(conn, table, column):
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")
    for table, column in OBSOLETE_COLUMNS:
        if _column_exists(conn, table, column):
            try:
                conn.execute(f"ALTER TABLE {table} DROP COLUMN {column}")
            except sqlite3.OperationalError:
                pass  # SQLite older than 3.35 can't drop columns; the unused column is harmless
    conn.commit()