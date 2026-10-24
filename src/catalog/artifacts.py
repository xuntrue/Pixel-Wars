import sqlite3

# --- CREATE ---
def create(conn: sqlite3.Connection, name: str) -> int:
    with conn:
        cur = conn.execute("INSERT INTO artifacts (name) VALUES (?)", (name,))
    return cur.lastrowid

# --- GET ---
def list_all(conn):
    return conn.execute("SELECT * FROM artifacts ORDER BY name").fetchall()

# --- PUT ---
def rename(conn, artifact_id: int, name: str) -> None:
    with conn:
        conn.execute("UPDATE artifacts SET name=? WHERE id=?", (name, artifact_id))

# --- DELETE ---
def delete(conn, artifact_id: int) -> None:
    """Raises sqlite3.IntegrityError if used in any recorded battle."""
    with conn:
        conn.execute("DELETE FROM artifacts WHERE id = ?", (artifact_id,))