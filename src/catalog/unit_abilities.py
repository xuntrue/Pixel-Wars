import sqlite3

# --- CREATE ---

# --- GET ---
def ability_ids_for_unit(conn, unit_id: int) -> set[int]:
    rows = conn.execute(
        "SELECT ability_id FROM unit_abilities WHERE unit_id = ?", (unit_id,)
    ).fetchall()
    return {r["ability_id"] for r in rows}


def set_for_unit(conn, unit_id: int, ability_ids) -> None:
    """ Replace the unit's abilities with exactly these (0 or more), in one transaction """
    with conn:
        conn.execute("DELETE FROM unit_abilities WHERE unit_id = ?", (unit_id,))
        conn.executemany(
            "INSERT INTO unit_abilities (unit_id, ability_id) VALUES (?, ?)",
            [(unit_id, a) for a in sorted(set(ability_ids))],
        )

def names_by_unit(conn) -> dict[int, list[str]]:
    """ unit_id -> sorted ability names, for every unit that has at least one """
    rows = conn.execute(
        """
        SELECT ua.unit_id, a.name
        FROM unit_abilities ua
        JOIN abilities a ON a.id = ua.ability_id
        ORDER BY a.name
        """
    ).fetchall()
    result: dict[int, list[str]] = {}
    for r in rows:
        result.setdefault(r["unit_id"], []).append(r["name"])
    return result

# --- PUT ---

# --- DELETE ---