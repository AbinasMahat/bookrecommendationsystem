import pickle
import sqlite3
import os

print("=" * 50)
print("SASRec Debug")
print("=" * 50)

# 1. Mapping files
for f in ['sasrec_isbn_to_id.pkl', 'sasrec_id_to_isbn.pkl', 'sasrec_id_to_title.pkl', 'sasrec_model.pt']:
    print(f"{f}: {'EXISTS' if os.path.exists(f) else 'MISSING'}")

isbn_to_id = pickle.load(open('sasrec_isbn_to_id.pkl', 'rb'))
id_to_title = pickle.load(open('sasrec_id_to_title.pkl', 'rb'))

print(f"\nMapped items: {len(isbn_to_id)}")
print("Sample map:", list(isbn_to_id.items())[:3])
print("Sample titles:", list(id_to_title.items())[:3])

# 2. Ratings from DB
db_path = 'instance/brs_database.db' if os.path.exists('instance/brs_database.db') else 'brs_database.db'
print(f"\nDB path: {db_path}")

conn = sqlite3.connect(db_path)
rows = conn.execute("SELECT id, user_id, isbn, book_title, rating FROM rating").fetchall()
print(f"Total ratings in DB: {len(rows)}")

matched = 0
for r in rows:
    isbn = str(r[2]).strip()
    title = str(r[3]).strip()
    by_isbn = isbn in isbn_to_id
    by_title = any(title.lower() == str(t).lower() for t in id_to_title.values())
    if by_isbn or by_title:
        matched += 1
    print(f"  user={r[1]} isbn=[{isbn}] title=[{title[:35]}] isbn_match={by_isbn} title_match={by_title}")

print(f"\nMatched ratings: {matched}/{len(rows)}")
conn.close()