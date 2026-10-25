import sqlite3

from src.api.database import ATTACK_SPEEDS

def _check_attack_speed(value) -> None:
    if value not in ATTACK_SPEEDS:
        raise ValueError(f"Attack speed must be one of: {', '.join(ATTACK_SPEEDS)}")

# --- CREATE ---
def create(conn: sqlite3.Connection, name: str, damage: float, speed: float, health: float, attack_speed: str) -> int:
    _check_attack_speed(attack_speed)
    with conn:
        cur = conn.execute(
            "INSERT INTO units (name, damage, speed, attack_speed, health) "
            "VALUES (?, ?, ?, ?, ?)",
            (name, damage, speed, attack_speed, health),
        )
    return cur.lastrowid

# --- GET ---
def get(conn, unit_id: int):
    return conn.execute("SELECT * FROM units WHERE id = ?", (unit_id,)).fetchone()

def list_all(conn):
    return conn.execute("SELECT * FROM units ORDER BY name").fetchall()

def battle_count(conn, unit_id: int) -> int:
    """ Number of recorded battles this unit appears in """
    return conn.execute(
        "SELECT COUNT(DISTINCT battle_id) FROM battle_units WHERE unit_id = ?",
        (unit_id,),
    ).fetchone()[0]

# --- PUT ---
def update(conn, unit_id: int, name: str, damage: float, speed: float, health: float, attack_speed: str) -> None:
    _check_attack_speed(attack_speed)
    with conn:
        conn.execute(
            "UPDATE units SET name=?, damage=?, speed=?, attack_speed=?, health=? WHERE id=?",
            (name, damage, speed, attack_speed, health, unit_id),
        )

# --- DELETE ---
def delete(conn, unit_id: int) -> None:
    """ Raises sqlite3.IntegrityError if the unit appears in any recorded battle """
    with conn:
        conn.execute("DELETE FROM units WHERE id = ?", (unit_id,))