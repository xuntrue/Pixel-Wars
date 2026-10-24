import sqlite3

# --- CREATE ---
def create(conn: sqlite3.Connection, name: str, damage: float,
           speed: float, health: float) -> int:
    with conn:
        cur = conn.execute(
            "INSERT INTO units (name, damage, speed, health) VALUES (?, ?, ?, ?)",
            (name, damage, speed, health),
        )
    return cur.lastrowid

# --- GET ---
def get(conn, unit_id: int):
    return conn.execute("SELECT * FROM units WHERE id = ?", (unit_id,)).fetchone()

def list_all(conn):
    return conn.execute("SELECT * FROM units ORDER BY name").fetchall()

# --- PUT ---
def update(conn, unit_id: int, name: str, damage: float,
           speed: float, health: float) -> None:
    with conn:
        conn.execute(
            "UPDATE units SET name=?, damage=?, speed=?, health=? WHERE id=?",
            (name, damage, speed, health, unit_id),
        )

# --- DELETE ---
def delete(conn, unit_id: int) -> None:
    """Raises sqlite3.IntegrityError if the unit appears in any recorded battle."""
    with conn:
        conn.execute("DELETE FROM units WHERE id = ?", (unit_id,))