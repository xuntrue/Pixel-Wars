import sqlite3

# --- CREATE ---
def create(conn: sqlite3.Connection, name: str) -> int:
    with conn:
        cur = conn.execute("INSERT INTO abilities (name) VALUES (?)", (name,))
    return cur.lastrowid

# --- GET ---
def list_all(conn):
    return conn.execute("SELECT * FROM abilities ORDER BY name").fetchall()

def unit_count(conn, ability_id: int) -> int:
    """ How many units currently have this ability """
    return conn.execute(
        "SELECT COUNT(*) FROM unit_abilities WHERE ability_id = ?", (ability_id,)
    ).fetchone()[0]

# --- PUT ---
def rename(conn, ability_id: int, name: str) -> None:
    with conn:
        conn.execute("UPDATE abilities SET name=? WHERE id=?", (name, ability_id))

# --- DELETE ---
def delete(conn, ability_id: int) -> None:
    """Also removes the ability from every unit that had it."""
    with conn:
        conn.execute("DELETE FROM unit_abilities WHERE ability_id = ?", (ability_id,))
        conn.execute("DELETE FROM abilities WHERE id = ?", (ability_id,))