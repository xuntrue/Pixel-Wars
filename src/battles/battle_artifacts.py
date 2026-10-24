import sqlite3

# --- CREATE ---
def add(conn, battle_id: int, side: str, artifact_id: int) -> None:
    conn.execute(
        "INSERT INTO battle_artifacts (battle_id, side, artifact_id) VALUES (?, ?, ?)",
        (battle_id, side, artifact_id),
    )

# --- GET ---
def for_battle(conn, battle_id: int):
    return conn.execute(
        "SELECT * FROM battle_artifacts WHERE battle_id = ?", (battle_id,)
    ).fetchall()

# --- PUT ---
# --- DELETE ---