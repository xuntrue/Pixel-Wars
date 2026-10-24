import sqlite3

from . import battle_units, battle_artifacts

# --- CREATE ---
def record(conn, winner: str,
           army_a: dict[int, int], army_b: dict[int, int],
           artifacts_a: list[int] = (), artifacts_b: list[int] = (),
           carry_over: bool = False, notes: str | None = None) -> int:
    """army_x maps unit_id -> count. Everything is saved in one transaction."""
    with conn:  # rolls back everything if any insert fails
        cur = conn.execute(
            "INSERT INTO battles (winner, carry_over, notes) VALUES (?, ?, ?)",
            (winner, int(carry_over), notes),
        )
        battle_id = cur.lastrowid
        for side, army, arts in (("A", army_a, artifacts_a), ("B", army_b, artifacts_b)):
            for unit_id, count in army.items():
                if count > 0:
                    battle_units.add(conn, battle_id, side, unit_id, count)
            for artifact_id in arts:
                battle_artifacts.add(conn, battle_id, side, artifact_id)
    return battle_id

# --- GET ---
def get_details(conn, battle_id: int) -> dict | None:
    """ Full picture of one battle: metadata, units per side, artifacts per side """
    battle = conn.execute(
        "SELECT * FROM battles WHERE id = ?", (battle_id,)
    ).fetchone()
    if battle is None:
        return None

    details = {
        "id": battle["id"],
        "timestamp": battle["timestamp"],
        "winner": battle["winner"],
        "carry_over": bool(battle["carry_over"]),
        "notes": battle["notes"],
        "sides": {"A": {"units": [], "artifacts": []},
                  "B": {"units": [], "artifacts": []}},
    }

    unit_rows = conn.execute(
        """
        SELECT bu.side, u.name, bu.count, u.damage, u.speed, u.health
        FROM battle_units bu
        JOIN units u ON u.id = bu.unit_id
        WHERE bu.battle_id = ?
        ORDER BY u.name
        """,
        (battle_id,),
    ).fetchall()
    for r in unit_rows:
        details["sides"][r["side"]]["units"].append({
            "name": r["name"], "count": r["count"],
            "damage": r["damage"], "speed": r["speed"], "health": r["health"],
        })

    artifact_rows = conn.execute(
        """
        SELECT ba.side, a.name
        FROM battle_artifacts ba
        JOIN artifacts a ON a.id = ba.artifact_id
        WHERE ba.battle_id = ?
        ORDER BY a.name
        """,
        (battle_id,),
    ).fetchall()
    for r in artifact_rows:
        details["sides"][r["side"]]["artifacts"].append(r["name"])

    return details

def format_details(details: dict) -> str:
    """Human-readable version of get_details() output."""
    lines = [
        f"Battle #{details['id']}  ({details['timestamp']})",
        f"Winner: {details['winner']}   Carry-over: {details['carry_over']}",
    ]
    if details["notes"]:
        lines.append(f"Notes: {details['notes']}")

    for side in ("A", "B"):
        s = details["sides"][side]
        total = sum(u["count"] for u in s["units"])
        lines.append(f"\nArmy {side} ({total} units)")
        for u in s["units"]:
            lines.append(
                f"  {u['count']:>3} x {u['name']}  "
                f"(dmg {u['damage']}, spd {u['speed']}, hp {u['health']})"
            )
        if not s["units"]:
            lines.append("  (no units)")
        lines.append("  Artifacts: " + (", ".join(s["artifacts"]) or "none"))

    return "\n".join(lines)

def list_all(conn):
    return conn.execute("SELECT * FROM battles ORDER BY id DESC").fetchall()

# --- PUT ---

# --- DELETE ---
def delete(conn, battle_id: int) -> None:
    """Cascades to battle_units and battle_artifacts."""
    with conn:
        conn.execute("DELETE FROM battles WHERE id = ?", (battle_id,))