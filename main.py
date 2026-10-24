from src.api.database import get_connection, init_db
from src.catalog import units, artifacts
from src.battles import battles

conn = get_connection()

init_db(conn)

# Create Units
knight = units.create(conn, "Knight", damage=10, speed=1.0, health=100)
hunter = units.create(conn, "Hunter", damage=15, speed=1.5, health=50)

# Create Artifacts
shield = artifacts.create(conn, "Shield")

# Create battle
bid = battles.record(conn, winner="B", army_a={knight: 3}, army_b={hunter: 4}, artifacts_a=[], artifacts_b=[shield])

print(dict(battles.list_all(conn)[0]))
print(battles.format_details(battles.get_details(conn, bid)))