import sqlite3

# --- CREATE ---
def add(conn, battle_id: int, side: str, unit_id: int,
        count: int, remaining: int, damage_dealt: float) -> None:
    conn.execute(
        "INSERT INTO battle_units "
        "(battle_id, side, unit_id, count, remaining, damage_dealt) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (battle_id, side, unit_id, count, remaining, damage_dealt),
    )
# --- GET ---
def for_battle(conn, battle_id: int):
    return conn.execute(
        "SELECT * FROM battle_units WHERE battle_id = ?", (battle_id,)
    ).fetchall()

# --- PUT ---

# --- DELETE ---
