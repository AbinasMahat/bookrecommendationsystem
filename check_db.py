import sqlite3
import os

# Check both possible locations
paths = [
    'brs_database.db',
    'instance/brs_database.db'
]

for db_path in paths:
    print(f"\nChecking: {db_path}")
    print("Exists:", os.path.exists(db_path))
    
    if os.path.exists(db_path):
        print("Full path:", os.path.abspath(db_path))
        conn = sqlite3.connect(db_path)
        c = conn.cursor()
        tables = c.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
        print("Tables:", tables)
        
        if tables:
            for t in tables:
                print(f"\n--- {t[0]} ---")
                rows = c.execute(f"SELECT * FROM {t[0]}").fetchall()
                print("Rows:", rows[:5] if rows else "Empty")
        conn.close()