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

def _winner_error(winner: str, survivors_a: int, survivors_b: int) -> str | None:
    """
    Rule for decisive battles: the winning army keeps at least 1 unit and the
    losing army has none. Returns an error message, or None if the result is
    consistent. Draws are not checked.
    """
    if winner not in ("A", "B"):
        return None
    loser = "B" if winner == "A" else "A"
    left = {"A": survivors_a, "B": survivors_b}
    if left[winner] < 1:
        return (f"Army {winner} is marked as the winner but has no survivors. "
                f"The winning army needs at least 1 surviving unit.")
    if left[loser] != 0:
        return (f"Army {loser} lost but still has {left[loser]} survivor(s). "
                f"The losing army must have 0 survivors.")
    return None

# --- CREATE ---
def record(conn, winner: str,
           army_a: dict[int, int], army_b: dict[int, int],
           remaining_a: dict[int, int], remaining_b: dict[int, int],
           damage_dealt_a: dict[int, float], damage_dealt_b: dict[int, float],
           artifacts_a: list[int] = (), artifacts_b: list[int] = (),
           notes: str | None = None) -> int:
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

    error = _winner_error(winner, sum(remaining_a.values()), sum(remaining_b.values()))
    if error:
        raise ValueError(error)

    with conn:  # rolls back everything if any insert fails
        cur = conn.execute(
            "INSERT INTO battles (winner, notes) VALUES (?, ?)",
            (winner, notes),
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
def count(conn) -> int:
    return conn.execute("SELECT COUNT(*) FROM battles").fetchone()[0]

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
        f"Winner: {details['winner']}",
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

def delete_all(conn) -> int:
    """
    Delete every battle with its unit and artifact rows. Units and artifacts
    themselves are kept. Returns how many battles were deleted.
    """
    deleted = count(conn)
    with conn:  # one transaction: all of it, or none of it
        conn.execute("DELETE FROM battle_units")
        conn.execute("DELETE FROM battle_artifacts")
        conn.execute("DELETE FROM battles")
    return deleted