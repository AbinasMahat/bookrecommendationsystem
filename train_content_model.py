"""
Content-Based Filtering Model
Uses Title + Author + Publisher with TF-IDF + Cosine Similarity
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print(" Content-Based Model Training Started")
print(f" Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 60)

# ====================== LOAD BOOKS ======================
print("\n[1/4] Loading Books data...")

try:
    books = pd.read_csv('Books.csv', low_memory=False)
except:
    books = pd.read_csv('books.csv', low_memory=False)

# Clean column names
books.columns = books.columns.str.strip()

# Standardize column names if needed
if 'Book-Title' not in books.columns and 'Title' in books.columns:
    books = books.rename(columns={
        'Title': 'Book-Title',
        'Author': 'Book-Author',
        'Year': 'Year-Of-Publication',
        'Publisher': 'Publisher'
    })

print(f"   Total books loaded: {books.shape[0]}")

# Keep only needed columns
books = books[['ISBN', 'Book-Title', 'Book-Author', 'Year-Of-Publication', 'Publisher']].copy()
books = books.dropna(subset=['Book-Title'])
books = books.drop_duplicates(subset=['Book-Title']).reset_index(drop=True)

# Fill missing values
books['Book-Author'] = books['Book-Author'].fillna('Unknown')
books['Publisher'] = books['Publisher'].fillna('Unknown')
books['Year-Of-Publication'] = books['Year-Of-Publication'].fillna(0)

print(f"   After cleaning: {books.shape[0]} unique books")

# ====================== CREATE CONTENT FEATURES ======================
print("\n[2/4] Creating content features...")

books['content'] = (
    books['Book-Title'].astype(str) + ' ' +
    books['Book-Author'].astype(str) + ' ' +
    books['Publisher'].astype(str)
)

books['content'] = books['content'].str.lower()

print("   Content text created (Title + Author + Publisher)")

# ====================== TF-IDF ======================
print("\n[3/4] Building TF-IDF matrix...")

tfidf = TfidfVectorizer(
    max_features=5000,
    stop_words='english',
    ngram_range=(1, 2)
)

tfidf_matrix = tfidf.fit_transform(books['content'])
print(f"   TF-IDF matrix shape: {tfidf_matrix.shape}")

# ====================== SIMILARITY ======================
print("\n[4/4] Calculating Content Similarity...")

# We will not store full similarity matrix (too big).
# Instead we store the TF-IDF matrix + books + vectorizer
# Similarity will be calculated on-the-fly in app.py

print("   Saving models...")

pickle.dump(books, open('content_books.pkl', 'wb'))
pickle.dump(tfidf, open('tfidf_vectorizer.pkl', 'wb'))
pickle.dump(tfidf_matrix, open('tfidf_matrix.pkl', 'wb'))

print("\n" + "=" * 60)
print(" CONTENT-BASED MODEL TRAINING COMPLETED!")
print("=" * 60)
print("\nSaved files:")
print("  • content_books.pkl")
print("  • tfidf_vectorizer.pkl")
print("  • tfidf_matrix.pkl")
print("\nYou can now update app.py")