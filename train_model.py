"""
Book Recommendation System - Advanced Training Script
-----------------------------------------------------
Algorithms:
1. SVD Matrix Factorization (Collaborative Filtering)
2. Hybrid Recommender (SVD + Weighted Popularity)
"""

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
import pickle
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

print("=" * 70)
print(" Book Recommendation System - ADVANCED MODEL TRAINING")
print(f" Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 70)

# ====================== 1. LOAD DATA ======================
print("\n[1/7] Loading datasets...")

try:
    books = pd.read_csv('Books.csv', low_memory=False)
    users = pd.read_csv('Users.csv')
    ratings = pd.read_csv('Ratings.csv')
except FileNotFoundError:
    books = pd.read_csv('books.csv', low_memory=False)
    users = pd.read_csv('users.csv')
    ratings = pd.read_csv('ratings.csv')

print(f"   Books   : {books.shape}")
print(f"   Users   : {users.shape}")
print(f"   Ratings : {ratings.shape}")

# ====================== 2. PREPROCESS ======================
print("\n[2/7] Preprocessing...")

# Books.csv in this workspace uses the compact Goodreads schema
# (Title/Author/Year/Publisher), while the runtime uses Book-Crossing names.
books.columns = books.columns.str.strip()
books = books.rename(columns={
    'Title': 'Book-Title',
    'Author': 'Book-Author',
    'Year': 'Year-Of-Publication',
})
required_columns = ['ISBN', 'Book-Title', 'Book-Author', 'Year-Of-Publication', 'Publisher']
missing_columns = [column for column in required_columns if column not in books.columns]
if missing_columns:
    raise ValueError(f"Books.csv is missing required columns: {missing_columns}")
for optional_column in ['Image-URL-S', 'Image-URL-M', 'Image-URL-L']:
    if optional_column not in books.columns:
        books[optional_column] = ''

books['ISBN'] = books['ISBN'].astype(str).str.strip()
books['Book-Title'] = books['Book-Title'].astype(str).str.strip()
books['Book-Author'] = books['Book-Author'].fillna('Unknown').astype(str).str.strip()
books = books[books['Book-Title'].ne('') & books['Book-Title'].ne('nan')]

ratings.columns = ratings.columns.str.strip()
if 'ISBN' not in ratings.columns or 'Book-Rating' not in ratings.columns:
    raise ValueError('Ratings.csv must contain ISBN and Book-Rating columns')
ratings['ISBN'] = ratings['ISBN'].astype(str).str.strip()
ratings['Book-Rating'] = pd.to_numeric(ratings['Book-Rating'], errors='coerce').fillna(0)

ratings_with_name = ratings.merge(books, on='ISBN')
ratings_with_name = ratings_with_name.dropna(subset=['Book-Title', 'Book-Author'])

# Keep only positive ratings (optional but improves quality)
ratings_with_name = ratings_with_name[ratings_with_name['Book-Rating'] > 0]

print(f"   Cleaned ratings: {ratings_with_name.shape}")

# ====================== 3. POPULARITY + WEIGHTED ======================
print("\n[3/7] Building Popularity & Weighted scores...")

num_rating_df = ratings_with_name.groupby('Book-Title').count()['Book-Rating'].reset_index()
num_rating_df.rename(columns={'Book-Rating': 'num_ratings'}, inplace=True)

avg_rating_df = ratings_with_name.groupby('Book-Title')['Book-Rating'].mean().reset_index()
avg_rating_df.rename(columns={'Book-Rating': 'avg_ratings'}, inplace=True)

popular_df = num_rating_df.merge(avg_rating_df, on='Book-Title')

# Top popular (for display)
N = 100
popular_df = (
    popular_df[popular_df['num_ratings'] >= 150]
    .sort_values('avg_ratings', ascending=False)
    .head(N)
)

books_unique = books.drop_duplicates('Book-Title')
popular_df = popular_df.merge(
    books_unique[['Book-Title', 'Book-Author', 'Image-URL-M', 'ISBN']],
    on='Book-Title', how='left'
).drop_duplicates('Book-Title')

# Full base for weighted
base_popular_df = num_rating_df.merge(avg_rating_df, on='Book-Title')
base_popular_df = base_popular_df.merge(
    books_unique[['Book-Title', 'Book-Author', 'Image-URL-M', 'ISBN']],
    on='Book-Title', how='left'
).drop_duplicates('Book-Title')

# IMDB Weighted Rating
C = base_popular_df['avg_ratings'].mean()
m = base_popular_df['num_ratings'].quantile(0.90)

base_popular_df['weighted_score'] = (
    (base_popular_df['num_ratings'] / (base_popular_df['num_ratings'] + m)) * base_popular_df['avg_ratings'] +
    (m / (base_popular_df['num_ratings'] + m)) * C
)

print(f"   Popular books kept : {len(popular_df)}")
print(f"   Weighted mean (C)  : {C:.3f}")
print(f"   Vote threshold (m) : {m:.0f}")

# ====================== 4. COLLABORATIVE FILTERING DATA ======================
print("\n[4/7] Preparing Collaborative Filtering data...")

# Active users
user_counts = ratings_with_name.groupby('User-ID').count()['Book-Rating']
active_users = user_counts[user_counts >= 100].index          # slightly relaxed
filtered_rating = ratings_with_name[ratings_with_name['User-ID'].isin(active_users)]

# Famous books
book_counts = filtered_rating.groupby('Book-Title').count()['Book-Rating']
famous_books = book_counts[book_counts >= 40].index
final_ratings = filtered_rating[filtered_rating['Book-Title'].isin(famous_books)]

print(f"   Active users       : {len(active_users)}")
print(f"   Famous books       : {len(famous_books)}")
print(f"   Final ratings used : {final_ratings.shape[0]}")

# Pivot table
pt = final_ratings.pivot_table(
    index='Book-Title',
    columns='User-ID',
    values='Book-Rating'
).fillna(0)

print(f"   Pivot table shape  : {pt.shape}")

# ====================== 5. SVD MATRIX FACTORIZATION (Algorithm 1) ======================
print("\n[5/7] Training SVD Matrix Factorization model...")

# We use TruncatedSVD on the user-item matrix (transposed for item factors)
n_components = 50   # latent factors (can be 30-100)

svd = TruncatedSVD(n_components=n_components, random_state=42)
book_factors = svd.fit_transform(pt)          # shape: (n_books, n_components)

# Reconstruct approximate ratings
reconstructed = np.dot(book_factors, svd.components_)

# Item-Item similarity from latent factors (much better than raw cosine)
similarity_scores = cosine_similarity(book_factors)

print(f"   SVD components     : {n_components}")
print(f"   Explained variance : {svd.explained_variance_ratio_.sum():.3f}")
print(f"   Similarity matrix  : {similarity_scores.shape}")

# ====================== 6. HYBRID SUPPORT DATA ======================
print("\n[6/7] Preparing Hybrid scoring data...")

# Create a lookup for weighted score
weighted_lookup = dict(zip(base_popular_df['Book-Title'], base_popular_df['weighted_score']))

# Normalize weighted scores to 0-1 for hybrid
max_w = base_popular_df['weighted_score'].max()
min_w = base_popular_df['weighted_score'].min()
base_popular_df['weighted_norm'] = (base_popular_df['weighted_score'] - min_w) / (max_w - min_w + 1e-8)

weighted_norm_lookup = dict(zip(base_popular_df['Book-Title'], base_popular_df['weighted_norm']))

print("   Hybrid lookup tables ready.")

# ====================== 7. SAVE EVERYTHING ======================
print("\n[7/7] Saving all models...")

pickle.dump(popular_df, open('popular.pkl', 'wb'))
pickle.dump(base_popular_df, open('base_popular_df.pkl', 'wb'))
pickle.dump(pt, open('pt.pkl', 'wb'))
pickle.dump(books, open('books.pkl', 'wb'))
pickle.dump(similarity_scores, open('similarity_scores.pkl', 'wb'))   # now from SVD
pickle.dump(svd, open('svd_model.pkl', 'wb'))
pickle.dump(book_factors, open('book_factors.pkl', 'wb'))
pickle.dump(weighted_norm_lookup, open('weighted_norm_lookup.pkl', 'wb'))

# Light books version
books_light = books[['ISBN', 'Book-Title', 'Book-Author', 'Image-URL-M']].drop_duplicates('Book-Title')
pickle.dump(books_light, open('books_light.pkl', 'wb'))

print("\n" + "=" * 70)
print(" TRAINING COMPLETED SUCCESSFULLY!")
print("=" * 70)
print("\nSaved files:")
print("  • popular.pkl")
print("  • base_popular_df.pkl")
print("  • pt.pkl")
print("  • books.pkl")
print("  • similarity_scores.pkl     ← SVD-based similarity")
print("  • svd_model.pkl")
print("  • book_factors.pkl")
print("  • weighted_norm_lookup.pkl  ← for Hybrid")
print("  • books_light.pkl")
print("\nAlgorithms ready:")
print("  1. SVD Matrix Factorization")
print("  2. Hybrid (SVD + Weighted Popularity)")
print("=" * 70)