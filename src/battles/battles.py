from . import battle_units, battle_artifacts

def _validate_side(side: str, army: dict[int, int], remaining: dict[int, int], damage_dealt: dict[int, float]) -> None:
    for name, mapping in (("survivors", remaining), ("damage", damage_dealt)):
        unknown = set(mapping) - set(army)
        if unknown:
            raise ValueError(f"Side {side}: {name} listed for units not in the army: {sorted(unknown)}")
    for unit_id, count in army.items():
        left = remaining.get(unit_id, 0)
        if not 0 <= left <= count:
            raise ValueError(f"Side {side}: unit {unit_id} has {left} survivors but only {count} deployed")
    for unit_id, dealt in damage_dealt.items():
        if dealt < 0:
            raise ValueError(f"Side {side}: unit {unit_id} has negative damage dealt")

# --- CREATE ---
def record(conn, winner: str,
           army_a: dict[int, int], army_b: dict[int, int],
           remaining_a: dict[int, int], remaining_b: dict[int, int],
           damage_dealt_a: dict[int, float], damage_dealt_b: dict[int, float],
           artifacts_a: list[int] = (), artifacts_b: list[int] = (),
           carry_over: bool = False, notes: str | None = None) -> int:
    """
    army_x:           unit_id -> number deployed
    remaining_x:      unit_id -> number alive at the end (missing = 0)
    damage_dealt_x:   unit_id -> total damage dealt by that whole group (missing = 0)
    Everything is saved in one transaction.
    """
    army_a = {u: c for u, c in army_a.items() if c > 0}
    army_b = {u: c for u, c in army_b.items() if c > 0}
    _validate_side("A", army_a, remaining_a, damage_dealt_a)
    _validate_side("B", army_b, remaining_b, damage_dealt_b)

    with conn:  # rolls back everything if any insert fails
        cur = conn.execute(
            "INSERT INTO battles (winner, carry_over, notes) VALUES (?, ?, ?)",
            (winner, int(carry_over), notes),
        )
        battle_id = cur.lastrowid
        for side, army, left, dealt, arts in (
            ("A", army_a, remaining_a, damage_dealt_a, artifacts_a),
            ("B", army_b, remaining_b, damage_dealt_b, artifacts_b),
        ):
            for unit_id, count in army.items():
                battle_units.add(conn, battle_id, side, unit_id, count,
                                 left.get(unit_id, 0), dealt.get(unit_id, 0))
            for artifact_id in arts:
                battle_artifacts.add(conn, battle_id, side, artifact_id)
    return battle_id

# --- GET ---
def list_all(conn):
    return conn.execute("SELECT * FROM battles ORDER BY id DESC").fetchall()

def get_details(conn, battle_id: int) -> dict | None:
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
        "sides": {
            s: {"units": [], "artifacts": [], "survivors": None, "damage_dealt": None}
            for s in ("A", "B")
        },
    }

    unit_rows = conn.execute(
        """
        SELECT bu.side, u.name, bu.count, bu.remaining, bu.damage_dealt,
               u.damage, u.speed, u.health
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
            "remaining": r["remaining"], "damage_dealt": r["damage_dealt"],
            "damage": r["damage"], "speed": r["speed"], "health": r["health"],
        })

    # Side totals are sums over the unit groups (None if any group is unknown)
    for side in details["sides"].values():
        left = [u["remaining"] for u in side["units"]]
        dealt = [u["damage_dealt"] for u in side["units"]]
        side["survivors"] = None if any(x is None for x in left) else sum(left)
        side["damage_dealt"] = None if any(x is None for x in dealt) else sum(dealt)

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
    def fmt(value):
        return "?" if value is None else value

    lines = [
        f"Battle #{details['id']}  ({details['timestamp']})",
        f"Winner: {details['winner']}   Carry-over: {details['carry_over']}",
    ]
    if details["notes"]:
        lines.append(f"Notes: {details['notes']}")

    for side in ("A", "B"):
        s = details["sides"][side]
        total = sum(u["count"] for u in s["units"])
        lines.append(
            f"\nArmy {side}: {total} deployed, {fmt(s['survivors'])} survived, "
            f"{fmt(s['damage_dealt'])} total damage dealt"
        )
        for u in s["units"]:
            lines.append(
                f"  {u['count']:>3} x {u['name']}  "
                f"({fmt(u['remaining'])} left, {fmt(u['damage_dealt'])} dmg dealt)  "
                f"[dmg {u['damage']}, spd {u['speed']}, hp {u['health']}]"
            )
        if not s["units"]:
            lines.append("  (no units)")
        lines.append("  Artifacts: " + (", ".join(s["artifacts"]) or "none"))

    return "\n".join(lines)

# --- PUT ---

# --- DELETE ---
def delete(conn, battle_id: int) -> None:
    """Cascades to battle_units and battle_artifacts."""
    with conn:
        conn.execute("DELETE FROM battles WHERE id = ?", (battle_id,))