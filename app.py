from flask import Flask, render_template, request, session, redirect, url_for, flash, jsonify, abort
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
from functools import wraps
import pickle
import json
import numpy as np
import requests
import torch
import torch.nn as nn
from urllib.parse import unquote, quote
import os
import secrets
import uuid
import pandas as pd
from math import ceil
from datetime import datetime
from time import monotonic
from pathlib import Path
import subprocess
import sys
import zipfile
from xml.etree import ElementTree
try:
    from docx import Document
except ImportError:
    Document = None
from sklearn.metrics.pairwise import cosine_similarity

# ====================== APP CONFIG ======================
app = Flask(__name__)
app.secret_key = os.environ.get('BRS_SECRET_KEY') or secrets.token_hex(32)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///brs_database.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_FOLDER = BASE_DIR / 'static' / 'uploads' / 'covers'
REFERENCE_UPLOAD_FOLDER = BASE_DIR / 'static' / 'uploads' / 'reference'
UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)
REFERENCE_UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)
app.config['UPLOAD_FOLDER'] = str(UPLOAD_FOLDER)
app.config['MAX_CONTENT_LENGTH'] = 5 * 1024 * 1024
ALLOWED_IMAGE_EXTENSIONS = {'jpg', 'jpeg', 'png', 'webp', 'gif'}

recommendation_cache = {}
recommendation_cache_ttl = 300
content_recommendation_cache = {}
content_recommendation_cache_ttl = 300
content_similarity_cache = {}
content_similarity_cache_limit = 16
description_cache = {}
description_cache_ttl = 86400
_svd_contract_warning_shown = False
EVALUATION_CACHE_PATH = BASE_DIR / 'evaluation_metrics.json'


def normalize_image_url(value):
    """Return a browser-safe cover URL and use the local fallback for missing values."""
    if value is None or pd.isna(value) or not str(value).strip():
        return '/static/book-placeholder.svg'
    image_url = str(value).strip()
    if image_url.casefold() in {'nan', 'none', 'null', 'undefined'}:
        return '/static/book-placeholder.svg'
    if image_url.startswith('//'):
        return f'https:{image_url}'
    if image_url.startswith('http://'):
        return f'https://{image_url[7:]}'
    if image_url.startswith('https://') or image_url.startswith('/'):
        return image_url
    return '/static/book-placeholder.svg'


def recommendation_match_percent(score):
    """Convert a normalized recommendation score into a user-facing percentage."""
    numeric_score = float(score or 0)
    if not np.isfinite(numeric_score):
        numeric_score = 0.0
    return max(0, min(100, round(numeric_score * 100)))


def get_evaluation_metrics():
    """Read the latest evaluator output without making the dashboard run inference."""
    if not EVALUATION_CACHE_PATH.exists():
        return None
    try:
        with EVALUATION_CACHE_PATH.open(encoding='utf-8') as metrics_file:
            metrics = json.load(metrics_file)
        return metrics if isinstance(metrics, dict) else None
    except (OSError, ValueError, TypeError):
        return None


app.jinja_env.globals['normalize_image_url'] = normalize_image_url


@app.context_processor
def inject_csrf_token():
    token = session.get('_csrf_token')
    if not token:
        token = secrets.token_urlsafe(32)
        session['_csrf_token'] = token
    return {'csrf_token': token}


@app.before_request
def validate_csrf_token():
    if request.method != 'POST':
        return None
    token = request.form.get('_csrf_token', '')
    expected = session.get('_csrf_token', '')
    if not expected or not token or not secrets.compare_digest(token, expected):
        return jsonify({'error': 'Invalid or missing CSRF token.'}), 400
    return None

# ====================== DATABASE MODELS ======================
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password = db.Column(db.String(200), nullable=False)
    is_admin = db.Column(db.Boolean, default=False, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    ratings = db.relationship('Rating', backref='user', lazy=True)
    wishlists = db.relationship('Wishlist', backref='user', lazy=True)


class Book(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(300), nullable=False, index=True)
    author = db.Column(db.String(200), nullable=False)
    genre = db.Column(db.String(120), default='')
    description = db.Column(db.Text, default='')
    image_url = db.Column(db.String(500), default='')
    isbn = db.Column(db.String(40), unique=True, nullable=False, index=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False, index=True)
    is_admin_added = db.Column(db.Boolean, default=True, nullable=False)
    initial_rating = db.Column(db.Float, default=0.0, nullable=False)
    popularity_score = db.Column(db.Float, default=0.0, nullable=False)


class UserPreference(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), unique=True, nullable=False)
    favorite_titles = db.Column(db.Text, default='')
    onboarding_complete = db.Column(db.Boolean, default=False, nullable=False)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class Rating(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    isbn = db.Column(db.String(20), nullable=False)
    book_title = db.Column(db.String(300), nullable=False)
    rating = db.Column(db.Float, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)


class Wishlist(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    isbn = db.Column(db.String(20), nullable=False)
    book_title = db.Column(db.String(300), nullable=False)
    author = db.Column(db.String(200))
    image = db.Column(db.String(500))
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    __table_args__ = (db.UniqueConstraint('user_id', 'isbn', name='unique_wishlist'),)

class RecentlyViewed(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    isbn = db.Column(db.String(20), nullable=False)
    book_title = db.Column(db.String(300), nullable=False)
    author = db.Column(db.String(200))
    image = db.Column(db.String(500))
    viewed_at = db.Column(db.DateTime, default=datetime.utcnow)


class SearchHistory(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False, index=True)
    query = db.Column(db.String(300), nullable=False)
    searched_at = db.Column(db.DateTime, default=datetime.utcnow, index=True)


class AdminDocument(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    original_filename = db.Column(db.String(300), nullable=False)
    stored_path = db.Column(db.String(500), nullable=False)
    extracted_text = db.Column(db.Text, nullable=False, default='')
    uploaded_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)
    uploaded_by = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)

    # ====================== SASRec MODEL CLASS ======================
class SASRec(nn.Module):
    def __init__(self, num_items, max_len=20, hidden_units=64, num_heads=2, num_blocks=2, dropout_rate=0.2):
        super(SASRec, self).__init__()
        self.num_items = num_items
        self.max_len = max_len
        self.item_emb = nn.Embedding(num_items + 1, hidden_units, padding_idx=0)
        self.pos_emb = nn.Embedding(max_len, hidden_units)
        self.emb_dropout = nn.Dropout(dropout_rate)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_units, nhead=num_heads,
            dim_feedforward=hidden_units * 4, dropout=dropout_rate,
            activation="relu", batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_blocks)
        self.layer_norm = nn.LayerNorm(hidden_units)

    def generate_causal_mask(self, seq_len, device):
        return torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1).bool()

    def forward(self, input_seqs):
        batch_size, seq_len = input_seqs.shape
        device = input_seqs.device
        positions = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
        seq_emb = self.item_emb(input_seqs) + self.pos_emb(positions)
        seq_emb = self.emb_dropout(seq_emb)
        padding_mask = input_seqs == 0
        causal_mask = self.generate_causal_mask(seq_len, device)
        out = self.transformer(seq_emb, mask=causal_mask, src_key_padding_mask=padding_mask)
        return self.layer_norm(out)

    def recommend_next(self, input_seq, top_k=8, exclude_ids=None):
        self.eval()
        with torch.no_grad():
            seq_out = self.forward(input_seq)
            last_item_repr = seq_out[:, -1, :]
            all_book_embs = self.item_emb.weight[1:]
            scores = torch.matmul(last_item_repr, all_book_embs.T)
            if exclude_ids:
                for eid in exclude_ids:
                    if 1 <= eid <= scores.shape[1]:
                        scores[0, eid - 1] = -1e9
            top_scores, top_indices = torch.topk(scores, k=min(top_k, scores.shape[1]), dim=-1)
            return (top_indices + 1).cpu().numpy()[0], top_scores.cpu().numpy()[0]

# ====================== LOAD MODELS ======================

# ====================== BOOKSHUB MODELS (Jupyter trained) ======================
class SASRecModel(nn.Module):
    def __init__(self, num_items, embed_dim=64, num_heads=2, max_seq_len=10, dropout=0.2):
        super(SASRecModel, self).__init__()
        self.item_emb = nn.Embedding(num_items + 1, embed_dim, padding_idx=0)
        self.pos_emb = nn.Embedding(max_seq_len, embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads,
            dim_feedforward=embed_dim * 2, dropout=dropout, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.fc = nn.Linear(embed_dim, num_items + 1)

    def forward(self, input_seq):
        seq_len = input_seq.size(1)
        positions = torch.arange(seq_len, device=input_seq.device).unsqueeze(0).expand(input_seq.size(0), -1)
        x = self.item_emb(input_seq) + self.pos_emb(positions)
        x = self.transformer(x)
        return self.fc(x[:, -1, :])


hub_books_df = None
hub_tfidf = None
hub_tfidf_matrix = None
hub_svd_model = None
hub_sasrec_model = None
hub_isbn_to_idx = {}
hub_title_col = hub_author_col = hub_isbn_col = None

def load_bookshub_models():
    global hub_books_df, hub_tfidf, hub_tfidf_matrix, hub_svd_model
    global hub_sasrec_model, hub_isbn_to_idx
    global hub_title_col, hub_author_col, hub_isbn_col

    model_dir = None
    for d in ['production_models', 'advance_pro_plus_models']:
        candidate = BASE_DIR / d
        if candidate.exists():
            model_dir = candidate
            break
    if not model_dir:
        print("[Bookshub] Model folder not found")
        return

    try:
        with open(model_dir / 'books_data.pkl', "rb") as f:
            hub_books_df = pickle.load(f)
        with open(model_dir / 'tfidf_vectorizer.pkl', "rb") as f:
            hub_tfidf = pickle.load(f)
        with open(model_dir / 'tfidf_matrix.pkl', "rb") as f:
            hub_tfidf_matrix = pickle.load(f)

        if (model_dir / 'svd_model.pkl').exists():
            with open(model_dir / 'svd_model.pkl', "rb") as f:
                hub_svd_model = pickle.load(f)

        if (model_dir / 'sasrec_mappings.pkl').exists():
            with open(model_dir / 'sasrec_mappings.pkl', "rb") as f:
                mappings = pickle.load(f)
                hub_isbn_to_idx = mappings.get('isbn_to_idx', {})

            hub_sasrec_model = SASRecModel(num_items=len(hub_isbn_to_idx))
            if (model_dir / 'sasrec_weights.pt').exists():
                hub_sasrec_model.load_state_dict(
                    torch.load(model_dir / 'sasrec_weights.pt', map_location='cpu')
                )
                hub_sasrec_model.eval()

        # Detect columns
        hub_title_col = [c for c in hub_books_df.columns if 'title' in c.lower()][0]
        hub_author_col = [c for c in hub_books_df.columns if 'author' in c.lower()][0]
        hub_isbn_col = [c for c in hub_books_df.columns if 'isbn' in c.lower() or c.lower() == 'book_id'][0]

        print(f"[Bookshub] Models loaded from {model_dir} | books={len(hub_books_df)}")
    except Exception as e:
        print(f"[Bookshub] Load failed: {e}")

load_bookshub_models()

#-----------------------------Old .ipynb-----------------------------------#
def load_pickle_safely(filename):
    try:
        with open(BASE_DIR / filename, 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        print(f"[Warning] Could not load {filename}: {e}")
        return None


print("Loading recommendation models...")
popular_df = load_pickle_safely('popular.pkl')
base_popular_df = load_pickle_safely('base_popular_df.pkl')
pt = load_pickle_safely('pt.pkl')
books = load_pickle_safely('books.pkl')
similarity_scores = load_pickle_safely('similarity_scores.pkl')
weighted_norm_lookup = load_pickle_safely('weighted_norm_lookup.pkl')
print("Models loaded!" if popular_df is not None else "Some models failed to load.")

# ====================== LOAD MODELS for new data ======================
content_books = load_pickle_safely('content_books.pkl')
tfidf_vectorizer = load_pickle_safely('tfidf_vectorizer.pkl')
tfidf_matrix = load_pickle_safely('tfidf_matrix.pkl')

# SASRec
sasrec_model = None
sasrec_isbn_to_id = load_pickle_safely('sasrec_isbn_to_id.pkl')
sasrec_id_to_isbn = load_pickle_safely('sasrec_id_to_isbn.pkl')
sasrec_id_to_title = load_pickle_safely('sasrec_id_to_title.pkl')

try:
    if (BASE_DIR / 'sasrec_model.pt').exists():
        checkpoint = torch.load(BASE_DIR / 'sasrec_model.pt', map_location='cpu')
        sasrec_model = SASRec(
            num_items=checkpoint['num_items'],
            max_len=checkpoint['max_len'],
            hidden_units=checkpoint['hidden_units'],
            num_heads=2,
            num_blocks=2,
        )
        sasrec_model.load_state_dict(checkpoint['model_state'])
        sasrec_model.eval()
        print("SASRec model loaded!")
except Exception as e:
    print(f"SASRec load failed: {e}")

# ====================== HELPERS ======================
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            flash("Please login first to access this page.", "warning")
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function


def admin_required(f):
    @wraps(f)
    @login_required
    def decorated_function(*args, **kwargs):
        user = db.session.get(User, session['user_id'])
        if not user or not user.is_admin:
            abort(403)
        return f(*args, **kwargs)
    return decorated_function


def book_to_catalog_row(book):
    return {
        'ISBN': book.isbn,
        'Book-Title': book.title,
        'Book-Author': book.author,
        'Book-Genre': book.genre or '',
        'Description': book.description or '',
        'Image-URL-M': book.image_url or '',
    }


def get_admin_book(isbn='', title=''):
    if isbn:
        book = Book.query.filter_by(isbn=str(isbn).strip()).first()
        if book:
            return book
    if title:
        return Book.query.filter(db.func.lower(Book.title) == str(title).strip().casefold()).first()
    return None


def get_book_description(title, author):
    query = f"{title} {author}"
    cache_key = query.casefold()
    cached = description_cache.get(cache_key)
    if cached and monotonic() - cached['created_at'] < description_cache_ttl:
        return cached['description']
    url = f"https://www.googleapis.com/books/v1/volumes?q={query}"
    description = "No description available."
    try:
        response = requests.get(url, timeout=6).json()
        items = response.get('items', [])
        if items:
            description = items[0]['volumeInfo'].get('description', description)
    except Exception:
        pass
    description_cache[cache_key] = {'created_at': monotonic(), 'description': description}
    return description


def get_catalog_book(isbn='', title=''):
    """Resolve submitted book fields against the trusted catalog artifact."""
    admin_book = get_admin_book(isbn=isbn, title=title)
    if admin_book:
        return book_to_catalog_row(admin_book)
    if books is None:
        return None
    if isbn and 'ISBN' in books.columns:
        matches = books[books['ISBN'].astype(str).str.strip() == str(isbn).strip()]
        if not matches.empty:
            return matches.iloc[0]
    if title and 'Book-Title' in books.columns:
        matches = books[books['Book-Title'].astype(str).str.casefold() == str(title).strip().casefold()]
        if not matches.empty:
            return matches.iloc[0]
    return None


def has_user_rated(user_id, isbn):
    return Rating.query.filter_by(user_id=user_id, isbn=isbn).first() is not None


def get_user_ratings(user_id):
    return Rating.query.filter_by(user_id=user_id).order_by(Rating.created_at.desc()).all()


def is_in_wishlist(user_id, isbn):
    return Wishlist.query.filter_by(user_id=user_id, isbn=isbn).first() is not None


def get_user_wishlist(user_id):
    return Wishlist.query.filter_by(user_id=user_id).order_by(Wishlist.created_at.desc()).all()


def get_seen_titles(user_id):
    """Return titles already rated, saved, or opened so recommendations stay fresh."""
    rated_titles = {
        rating.book_title for rating in Rating.query.filter_by(user_id=user_id).all()
    }
    saved_titles = {
        item.book_title for item in Wishlist.query.filter_by(user_id=user_id).all()
    }
    viewed_titles = {
        item.book_title for item in RecentlyViewed.query.filter_by(user_id=user_id).all()
    }
    return rated_titles | saved_titles | viewed_titles


def get_user_preference_titles(user_id):
    preference = UserPreference.query.filter_by(user_id=user_id).first()
    if not preference or not preference.favorite_titles:
        return []
    return [title for title in preference.favorite_titles.split('|') if title]


def get_user_search_titles(user_id, limit=20):
    """Return distinct recommendation searches, newest first."""
    searches = db.session.query(SearchHistory).filter_by(user_id=user_id).order_by(
        SearchHistory.searched_at.desc()
    ).limit(limit).all()
    return list(dict.fromkeys(item.query for item in searches if item.query.strip()))


def record_search_history(user_id, query):
    query = str(query or '').strip()
    if not query:
        return
    db.session.add(SearchHistory(user_id=user_id, query=query[:300]))
    db.session.commit()


def get_popular_fallback(top_n=8, exclude_titles=None):
    """Provide a useful cold-start list when personalized models have no signal."""
    if base_popular_df is None:
        return []
    excluded = {str(title).casefold() for title in (exclude_titles or set())}
    fallback = base_popular_df.sort_values(
        ['weighted_score', 'avg_ratings'], ascending=False
    )
    results = []
    for _, row in fallback.iterrows():
        title = str(row.get('Book-Title', ''))
        if not title or title.casefold() in excluded:
            continue
        results.append({
            'title': title,
            'author': row.get('Book-Author', 'Unknown'),
            'image': row.get('Image-URL-M', ''),
            'isbn': row.get('ISBN', ''),
            'similarity_score': 0.5,
            'predicted_rating': round(float(row.get('avg_ratings', 0)), 2),
            'explanation': 'Popular with readers',
        })
        if len(results) >= top_n:
            break
    return results


def get_personalized_recommendations(user_id, top_n=8):
    """Generate recommendations from ratings, searches, and recent book views."""
    user_ratings = Rating.query.filter_by(user_id=user_id).all()
    preference_titles = get_user_preference_titles(user_id)
    viewed_titles = list(dict.fromkeys(
        item.book_title for item in RecentlyViewed.query.filter_by(user_id=user_id)
        .order_by(RecentlyViewed.viewed_at.desc()).limit(20).all()
    ))
    searched_titles = get_user_search_titles(user_id)
    behavior_titles = list(dict.fromkeys(searched_titles + viewed_titles))

    # Browsing alone becomes a useful signal after five distinct book views.
    has_profile_signal = bool(user_ratings or preference_titles or len(viewed_titles) >= 5)
    if not has_profile_signal:
        return []
    if pt is None or similarity_scores is None or books is None:
        return get_popular_fallback(top_n=top_n, exclude_titles=behavior_titles)

    seen_titles = get_seen_titles(user_id)

    liked_books = [r.book_title for r in user_ratings if r.rating >= 4]
    liked_books.extend(preference_titles + behavior_titles)
    if not liked_books:
        liked_books = [r.book_title for r in user_ratings]

    scored_books = {}

    for book_title in liked_books:
        matches = np.where(pt.index == book_title)[0]
        if len(matches) == 0:
            # Fixed: regex=False to avoid special character errors
            matches = np.where(pt.index.str.lower().str.contains(book_title.lower()[:30], na=False, regex=False))[0]
        
        if len(matches):
            idx = matches[0]
            similar = sorted(list(enumerate(similarity_scores[idx])), key=lambda x: x[1], reverse=True)[1:15]
            for sim_idx, score in similar:
                title = pt.index[sim_idx]
                if title not in seen_titles:
                    scored_books[title] = scored_books.get(title, 0) + float(score)
        elif 'get_content_based_recommendations' in globals():
            for candidate in get_content_based_recommendations(book_title, top_n=12):
                title = candidate['title']
                if title not in seen_titles:
                    scored_books[title] = scored_books.get(title, 0) + float(candidate.get('similarity_score', 0))

    if not scored_books and len(viewed_titles) >= 5:
        return get_popular_fallback(top_n=top_n, exclude_titles=seen_titles)

    # Sort by accumulated score
    sorted_books = sorted(scored_books.items(), key=lambda x: x[1], reverse=True)[:top_n]

    results = []
    for title, score in sorted_books:
        temp_df = books[books['Book-Title'] == title].drop_duplicates('Book-Title')
        if temp_df.empty:
            continue
        results.append({
            'title': title,
            'author': temp_df['Book-Author'].values[0],
            'image': temp_df['Image-URL-M'].values[0],
            'isbn': temp_df['ISBN'].values[0] if 'ISBN' in temp_df.columns else '',
            'similarity_score': round(score, 4),
            'predicted_rating': round(min(score * 2.5, 5.0), 2)
        })
    return results

# "class SASRec(nn.Module):
#     def __init__(self, num_items, max_len=20, hidden_units=64, num_heads=2, num_blocks=2, dropout_rate=0.2):
#         super(SASRec, self).__init__()
#         self.num_items = num_items
#         self.max_len = max_len
#         self.item_emb = nn.Embedding(num_items + 1, hidden_units, padding_idx=0)
#         self.pos_emb = nn.Embedding(max_len, hidden_units)
#         self.emb_dropout = nn.Dropout(dropout_rate)
#         encoder_layer = nn.TransformerEncoderLayer(
#             d_model=hidden_units, nhead=num_heads,
#             dim_feedforward=hidden_units * 4, dropout=dropout_rate,
#             activation="relu", batch_first=True,
#         )
#         self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_blocks)
#         self.layer_norm = nn.LayerNorm(hidden_units)

#     def generate_causal_mask(self, seq_len, device):
#         return torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1).bool()

#     def forward(self, input_seqs):
#         batch_size, seq_len = input_seqs.shape
#         device = input_seqs.device
#         positions = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
#         seq_emb = self.item_emb(input_seqs) + self.pos_emb(positions)
#         seq_emb = self.emb_dropout(seq_emb)
#         padding_mask = input_seqs == 0
#         causal_mask = self.generate_causal_mask(seq_len, device)
#         out = self.transformer(seq_emb, mask=causal_mask, src_key_padding_mask=padding_mask)
#         return self.layer_norm(out)

#     def recommend_next(self, input_seq, top_k=8, exclude_ids=None):
#         self.eval()
#         with torch.no_grad():
#             seq_out = self.forward(input_seq)
#             last_item_repr = seq_out[:, -1, :]
#             all_book_embs = self.item_emb.weight[1:]
#             scores = torch.matmul(last_item_repr, all_book_embs.T)
#             if exclude_ids:
#                 for eid in exclude_ids:
#                     if 1 <= eid <= scores.shape[1]:
#                         scores[0, eid - 1] = -1e9
#             top_scores, top_indices = torch.topk(scores, k=min(top_k, scores.shape[1]), dim=-1)
#             return (top_indices + 1).cpu().numpy()[0], top_scores.cpu().numpy()[0]
        
def get_bookshub_hybrid(query_title, user_id=1, top_n=8, w_tfidf=0.35, w_svd=0.35, w_sasrec=0.30):
    """Bookshub Multi-Algorithm Hybrid: TF-IDF + SVD + SASRec"""
    if hub_books_df is None or hub_tfidf_matrix is None:
        return []

    from sklearn.metrics.pairwise import linear_kernel

    matches = hub_books_df[
        hub_books_df[hub_title_col].astype(str).str.contains(query_title, case=False, na=False)
    ]
    if matches.empty:
        return []

    query_idx = matches.index[0]

    # Normalize weights
    total_w = w_tfidf + w_svd + w_sasrec
    if total_w <= 0:
        total_w = 1.0
    w_tfidf, w_svd, w_sasrec = w_tfidf/total_w, w_svd/total_w, w_sasrec/total_w

    # 1. TF-IDF
    score_tfidf = linear_kernel(hub_tfidf_matrix[query_idx], hub_tfidf_matrix).flatten()

    # Score a bounded candidate pool instead of iterating over every book.
    candidate_pool_size = min(400, len(hub_books_df))
    candidate_indices = np.argpartition(score_tfidf, -candidate_pool_size)[-candidate_pool_size:]

    # 2. Collaborative SVD similarity. The root trainer uses sklearn
    # TruncatedSVD, so do not call the incompatible Surprise .predict API.
    score_svd = np.zeros(len(hub_books_df))
    svd_query_matches = np.where(pt.index.astype(str).str.casefold() == str(query_title).casefold())[0] if pt is not None else []
    if len(svd_query_matches) and similarity_scores is not None:
        svd_query_idx = svd_query_matches[0]
        for idx in candidate_indices:
            candidate_title = str(hub_books_df.iloc[idx][hub_title_col])
            candidate_matches = np.where(pt.index.astype(str).str.casefold() == candidate_title.casefold())[0]
            if len(candidate_matches):
                score_svd[idx] = float(similarity_scores[svd_query_idx, candidate_matches[0]])

    # 3. SASRec
    score_sasrec = np.zeros(len(hub_books_df))
    if hub_sasrec_model is not None and hub_isbn_to_idx:
        query_isbn = hub_books_df.loc[query_idx, hub_isbn_col]
        mapped_idx = hub_isbn_to_idx.get(query_isbn, 0)
        input_seq = torch.tensor([[0]*9 + [mapped_idx]], dtype=torch.long)
        with torch.no_grad():
            logits = hub_sasrec_model(input_seq).squeeze(0)
            probs = torch.softmax(logits, dim=0).numpy()
            for idx in candidate_indices:
                row = hub_books_df.iloc[idx]
                b_idx = hub_isbn_to_idx.get(row[hub_isbn_col], 0)
                if b_idx < len(probs):
                    score_sasrec[idx] = probs[b_idx]

    final_scores = (w_tfidf * score_tfidf) + (w_svd * score_svd) + (w_sasrec * score_sasrec)
    top_indices = [i for i in candidate_indices[np.argsort(final_scores[candidate_indices])[::-1]]
                   if i != query_idx][:top_n]

    results = []
    for idx in top_indices:
        tot = final_scores[idx] if final_scores[idx] > 0 else 1.0
        p_tf = round((w_tfidf * score_tfidf[idx] / tot) * 100, 1)
        p_sv = round((w_svd * score_svd[idx] / tot) * 100, 1)
        p_sa = round((w_sasrec * score_sasrec[idx] / tot) * 100, 1)

        title = hub_books_df.loc[idx, hub_title_col]
        author = hub_books_df.loc[idx, hub_author_col]
        isbn = hub_books_df.loc[idx, hub_isbn_col]

        # Try get image from old books.pkl
        image = ''
        if books is not None:
            temp = books[books['Book-Title'] == title]
            if not temp.empty and 'Image-URL-M' in temp.columns:
                image = temp['Image-URL-M'].values[0]

        results.append({
            'title': title,
            'author': author,
            'image': image,
            'isbn': str(isbn),
            'similarity_score': round(float(final_scores[idx]), 4),
            'predicted_rating': round(min(float(final_scores[idx]) * 5, 5.0), 2),
            'explanation': f"TF-IDF: {p_tf}% | SVD: {p_sv}% | SASRec: {p_sa}%"
        })
    return results

def add_recently_viewed(user_id, isbn, title, author, image):
    if not isbn or not title:
        return
    # Delete if already exists (to update time)
    old = RecentlyViewed.query.filter_by(user_id=user_id, isbn=isbn).first()
    if old:
        db.session.delete(old)
    
    item = RecentlyViewed(
        user_id=user_id,
        isbn=isbn,
        book_title=title,
        author=author,
        image=image
    )
    db.session.add(item)
    db.session.commit()

    # Keep only last 12
    all_items = RecentlyViewed.query.filter_by(user_id=user_id).order_by(RecentlyViewed.viewed_at.desc()).all()
    if len(all_items) > 12:
        for extra in all_items[12:]:
            db.session.delete(extra)
        db.session.commit()


def get_recently_viewed(user_id, limit=8):
    return RecentlyViewed.query.filter_by(user_id=user_id)\
        .order_by(RecentlyViewed.viewed_at.desc()).limit(limit).all()

# ====================== AUTH ROUTES ======================
@app.route('/register', methods=['GET', 'POST'])
def register():
    if 'user_id' in session:
        return redirect(url_for('index'))

    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        email = request.form.get('email', '').strip().lower()
        password = request.form.get('password', '')
        confirm_password = request.form.get('confirm_password', '')

        if not username or not email or not password:
            flash("All fields are required.", "danger")
            return redirect(url_for('register'))

        if password != confirm_password:
            flash("Passwords do not match.", "danger")
            return redirect(url_for('register'))

        if len(password) < 6:
            flash("Password must be at least 6 characters.", "danger")
            return redirect(url_for('register'))

        if User.query.filter_by(username=username).first():
            flash("Username already exists.", "danger")
            return redirect(url_for('register'))

        if User.query.filter_by(email=email).first():
            flash("Email already registered.", "danger")
            return redirect(url_for('register'))

        hashed_password = generate_password_hash(password)
        admin_email = os.environ.get('BRS_ADMIN_EMAIL', 'abi2003nas@gmail.com').strip().lower()
        new_user = User(
            username=username,
            email=email,
            password=hashed_password,
            is_admin=email == admin_email,
        )
        db.session.add(new_user)
        db.session.commit()

        flash("Account created successfully! Please login.", "success")
        return redirect(url_for('login'))

    return render_template('register.html')


@app.route('/login', methods=['GET', 'POST'])
def login():
    if 'user_id' in session:
        return redirect(url_for('index'))

    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '')

        user = User.query.filter_by(username=username).first()

        if user and check_password_hash(user.password, password):
            session['user_id'] = user.id
            session['username'] = user.username
            session['is_admin'] = bool(user.is_admin)
            flash(f"Welcome back, {user.username}!", "success")
            preference = UserPreference.query.filter_by(user_id=user.id).first()
            if not Rating.query.filter_by(user_id=user.id).first() and not preference:
                return redirect(url_for('onboarding'))
            return redirect(url_for('index'))
        else:
            flash("Invalid username or password.", "danger")
            return redirect(url_for('login'))

    return render_template('login.html')


def save_cover_upload(upload):
    if not upload or not upload.filename:
        return ''
    extension = Path(secure_filename(upload.filename)).suffix.casefold().lstrip('.')
    if extension not in ALLOWED_IMAGE_EXTENSIONS:
        raise ValueError('Cover images must be JPG, PNG, WEBP, or GIF files.')
    filename = f'{uuid.uuid4().hex}.{extension}'
    upload.save(UPLOAD_FOLDER / filename)
    return f'/static/uploads/covers/{filename}'


def extract_word_document(upload):
    """Extract paragraphs and table cells from an uploaded DOCX reference file."""
    filename = secure_filename(upload.filename or '')
    if not filename.casefold().endswith('.docx'):
        raise ValueError('Please upload a .docx Word document.')
    if Document is not None:
        document = Document(upload.stream)
        sections = [paragraph.text.strip() for paragraph in document.paragraphs if paragraph.text.strip()]
        for table in document.tables:
            for row in table.rows:
                cells = [cell.text.strip() for cell in row.cells if cell.text.strip()]
                if cells:
                    sections.append(' | '.join(cells))
    else:
        upload.stream.seek(0)
        with zipfile.ZipFile(upload.stream) as archive:
            document_xml = archive.read('word/document.xml')
        root = ElementTree.fromstring(document_xml)
        text_nodes = [node.text.strip() for node in root.iter() if node.text and node.text.strip()]
        sections = [' '.join(text_nodes)] if text_nodes else []
    return filename, '\n'.join(sections)


@app.route('/admin', methods=['GET', 'POST'])
@admin_required
def admin_dashboard():
    edit_id = request.args.get('edit', type=int)
    editing_book = db.session.get(Book, edit_id) if edit_id else None
    if request.method == 'POST':
        title = request.form.get('title', '').strip()
        author = request.form.get('author', '').strip()
        isbn = request.form.get('isbn', '').strip()
        if not title or not author or not isbn:
            flash('Title, author, and ISBN/ID are required.', 'danger')
            return redirect(url_for('admin_dashboard', edit=editing_book.id if editing_book else None))
        duplicate = Book.query.filter(Book.isbn == isbn, Book.id != (editing_book.id if editing_book else 0)).first()
        if duplicate:
            flash('A book with that ISBN/ID already exists.', 'danger')
            return redirect(url_for('admin_dashboard'))
        try:
            image_url = request.form.get('image_url', '').strip()
            uploaded_image = save_cover_upload(request.files.get('cover'))
            image_url = uploaded_image or image_url
        except ValueError as error:
            flash(str(error), 'danger')
            return redirect(url_for('admin_dashboard', edit=editing_book.id if editing_book else None))
        book = editing_book or Book(is_admin_added=True)
        book.title = title
        book.author = author
        book.isbn = isbn
        book.genre = request.form.get('genre', '').strip()
        book.description = request.form.get('description', '').strip()
        if image_url:
            book.image_url = image_url
        if not book.image_url:
            book.image_url = '/static/book-placeholder.svg'
        db.session.add(book)
        db.session.commit()
        recommendation_cache.clear()
        content_recommendation_cache.clear()
        flash('Book updated.' if editing_book else 'Book added to the catalog.', 'success')
        return redirect(url_for('admin_dashboard'))
    books_added = Book.query.filter_by(is_admin_added=True).order_by(Book.created_at.desc()).all()
    reference_document = AdminDocument.query.order_by(AdminDocument.uploaded_at.desc()).first()
    return render_template('admin_dashboard.html', books=books_added, editing_book=editing_book,
                           reference_document=reference_document, username=session.get('username'))


@app.post('/admin/reference-document')
@admin_required
def upload_reference_document():
    upload = request.files.get('reference_document')
    if not upload or not upload.filename:
        flash('Choose a Word document to upload.', 'danger')
        return redirect(url_for('admin_dashboard'))
    try:
        filename, extracted_text = extract_word_document(upload)
        stored_name = f'{uuid.uuid4().hex}.docx'
        stored_path = REFERENCE_UPLOAD_FOLDER / stored_name
        upload.stream.seek(0)
        upload.save(stored_path)
    except (ValueError, OSError, Exception) as error:
        flash(f'Could not read that Word document: {error}', 'danger')
        return redirect(url_for('admin_dashboard'))

    previous = AdminDocument.query.order_by(AdminDocument.uploaded_at.desc()).all()
    for document in previous:
        old_path = BASE_DIR / document.stored_path.removeprefix('/static/')
        if old_path.exists():
            old_path.unlink()
        db.session.delete(document)
    db.session.add(AdminDocument(
        original_filename=filename,
        stored_path=f'/static/uploads/reference/{stored_name}',
        extracted_text=extracted_text,
        uploaded_by=session['user_id'],
    ))
    db.session.commit()
    flash('Reference Word document uploaded and read successfully.', 'success')
    return redirect(url_for('admin_dashboard'))


@app.post('/admin/books/<int:book_id>/delete')
@admin_required
def delete_admin_book(book_id):
    book = db.get_or_404(Book, book_id)
    if book.image_url and book.image_url.startswith('/static/uploads/covers/'):
        image_path = BASE_DIR / book.image_url.removeprefix('/static/')
        if image_path.exists():
            image_path.unlink()
    db.session.delete(book)
    db.session.commit()
    recommendation_cache.clear()
    content_recommendation_cache.clear()
    flash('Book removed from the catalog.', 'info')
    return redirect(url_for('admin_dashboard'))


@app.route('/onboarding', methods=['GET', 'POST'])
@login_required
def onboarding():
    if request.method == 'POST':
        submitted_titles = request.form.getlist('favorite_titles')[:5]
        valid_titles = []
        for title in submitted_titles:
            catalog_book = get_catalog_book(title=title.strip())
            if catalog_book is not None:
                valid_titles.append(str(catalog_book.get('Book-Title', title)).strip())
        if len(valid_titles) < 2:
            flash('Choose at least two books so we can personalize your recommendations.', 'warning')
            return redirect(url_for('onboarding'))

        preference = UserPreference.query.filter_by(user_id=session['user_id']).first()
        if preference is None:
            preference = UserPreference(user_id=session['user_id'])
            db.session.add(preference)
        preference.favorite_titles = '|'.join(dict.fromkeys(valid_titles))
        preference.onboarding_complete = True
        db.session.commit()
        flash('Your reading profile is ready. Recommendations are personalized now.', 'success')
        return redirect(url_for('index'))

    favorite_books = []
    if base_popular_df is not None:
        favorite_books = base_popular_df.sort_values(
            ['weighted_score', 'avg_ratings'], ascending=False
        ).head(24).to_dict('records')
    return render_template('onboarding.html', favorite_books=favorite_books,
                           username=session.get('username'))


@app.route('/forgot_password', methods=['GET', 'POST'])
def forgot_password():
    if 'user_id' in session:
        return redirect(url_for('index'))

    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        email = request.form.get('email', '').strip().lower()
        password = request.form.get('password', '')
        confirm_password = request.form.get('confirm_password', '')

        user = User.query.filter_by(username=username, email=email).first()
        if not user:
            flash('We could not verify those account details.', 'danger')
            return redirect(url_for('forgot_password'))
        if len(password) < 6:
            flash('Password must be at least 6 characters.', 'danger')
            return redirect(url_for('forgot_password'))
        if password != confirm_password:
            flash('Passwords do not match.', 'danger')
            return redirect(url_for('forgot_password'))

        user.password = generate_password_hash(password)
        db.session.commit()
        flash('Password updated successfully. Please login with your new password.', 'success')
        return redirect(url_for('login'))

    return render_template('forgot_password.html')


@app.route('/logout')
def logout():
    session.clear()
    flash("You have been logged out successfully.", "info")
    return redirect(url_for('landing'))


# ====================== LANDING ======================
@app.route('/')
def landing():
    if 'user_id' in session:
        return redirect(url_for('index'))
    return render_template('landing.html')


# ====================== HOME ======================
@app.route('/home')
@login_required
def index():
    method = request.args.get('method', 'weighted')
    try:
        m_value = min(0.99, max(0.01, float(request.args.get('m', 0.90))))
    except (TypeError, ValueError):
        m_value = 0.90
    try:
        page = max(1, int(request.args.get('page', 1)))
    except (TypeError, ValueError):
        page = 1
    per_page = 12

    if popular_df is None or base_popular_df is None:
        flash("Recommendation data not loaded.", "danger")
        return render_template('index.html', book_name=[], author=[], image=[], votes=[],
                               avg_rating=[], weighted_rating=[], rating=[],
                               user_id=session.get('user_id'), username=session.get('username'),
                               method=method, m_value=m_value, current_page=1, total_pages=1,
                               personalized=[])

    if method == 'weighted':
        df = base_popular_df.copy()
        C = df['avg_ratings'].mean()
        smoothing_threshold = df['num_ratings'].quantile(0.90)
        if m_value > 0.01:
            catalog_threshold = df['num_ratings'].quantile(m_value)
            df = df[df['num_ratings'] >= catalog_threshold]
        df['weighted_score'] = (
            (df['num_ratings'] / (df['num_ratings'] + smoothing_threshold)) * df['avg_ratings'] +
            (smoothing_threshold / (df['num_ratings'] + smoothing_threshold)) * C
        )
        top_df = df.sort_values(['weighted_score', 'avg_ratings'], ascending=False).head(120)
        top_df['display_rating'] = top_df['weighted_score']
    else:
        top_df = popular_df.head(120).copy()
        top_df['weighted_score'] = top_df['avg_ratings']
        top_df['display_rating'] = top_df['avg_ratings']

    total_books = len(top_df)
    total_pages = max(1, ceil(total_books / per_page))
    page = max(1, min(page, total_pages))
    start = (page - 1) * per_page
    end = min(start + per_page, total_books)
    paginated_df = top_df.iloc[start:end]

    return render_template(
        'index.html',
        book_name=list(paginated_df['Book-Title'].values),
        author=list(paginated_df['Book-Author'].values),
        image=list(paginated_df['Image-URL-M'].values),
        votes=list(paginated_df['num_ratings'].values),
        avg_rating=list(paginated_df['avg_ratings'].values),
        weighted_rating=list(paginated_df['weighted_score'].values),
        rating=list(paginated_df['display_rating'].values),
        user_id=session.get('user_id'),
        username=session.get('username'),
        method=method,
        m_value=m_value,
        current_page=page,
        total_pages=total_pages
    )


@app.route('/my_recommendations')
@login_required
def my_recommendations():
    personalized = get_personalized_recommendations(session['user_id'], top_n=8)
    recently_viewed = get_recently_viewed(session['user_id'], limit=8)
    return render_template(
        'my_recommendations.html',
        personalized=personalized,
        recently_viewed=recently_viewed,
        username=session.get('username')
    )

def get_content_based_recommendations(book_title, top_n=8):
    """Content-Based recommendations using TF-IDF"""
    if content_books is None or tfidf_matrix is None:
        return []

    book_title = str(book_title).strip()

    matches = content_books[content_books['Book-Title'].str.lower() == book_title.lower()]

    if matches.empty:
        matches = content_books[
            content_books['Book-Title'].str.lower().str.contains(
                book_title.lower()[:50], na=False, regex=False
            )
        ]

    if matches.empty:
        return []

    idx = matches.index[0]
    cached = content_recommendation_cache.get((book_title.casefold(), top_n))
    if cached and monotonic() - cached['created_at'] < content_recommendation_cache_ttl:
        return cached['data']
    cosine_sim = content_similarity_cache.get(idx)
    if cosine_sim is None:
        cosine_sim = cosine_similarity(tfidf_matrix[idx:idx+1], tfidf_matrix).flatten()
        content_similarity_cache[idx] = cosine_sim
        if len(content_similarity_cache) > content_similarity_cache_limit:
            content_similarity_cache.pop(next(iter(content_similarity_cache)))
    similar_indices = cosine_sim.argsort()[::-1][1:top_n + 5]

    results = []
    seen = set()

    for i in similar_indices:
        if len(results) >= top_n:
            break

        row = content_books.iloc[i]
        title = str(row['Book-Title'])

        if title in seen or title.lower() == book_title.lower():
            continue
        seen.add(title)

        image = ''
        isbn = str(row.get('ISBN', ''))

        if books is not None:
            temp = books[books['Book-Title'] == title]
            if not temp.empty:
                if 'Image-URL-M' in temp.columns:
                    image = temp['Image-URL-M'].values[0]
                if 'ISBN' in temp.columns and (not isbn or isbn == 'nan'):
                    isbn = str(temp['ISBN'].values[0])

        results.append({
            'title': title,
            'author': str(row.get('Book-Author', 'Unknown')),
            'image': image,
            'isbn': isbn,
            'similarity_score': round(float(cosine_sim[i]), 4),
            'predicted_rating': round(min(float(cosine_sim[i]) * 5, 5.0), 2)
        })

    content_recommendation_cache[(book_title.casefold(), top_n)] = {
        'created_at': monotonic(),
        'data': results,
    }
    return results
# ====================== RECOMMEND ======================
def get_collaborative_recommendations(book_title, top_n=20):
    """Return collaborative candidates without exposing model selection to users."""
    if pt is None or similarity_scores is None or books is None:
        return []

    matches = np.where(pt.index == book_title)[0]
    if len(matches) == 0:
        matches = np.where(pt.index.str.lower() == book_title.lower())[0]
    if len(matches) == 0:
        matches = np.where(
            pt.index.str.lower().str.contains(book_title.lower()[:40], na=False, regex=False)
        )[0]
    if len(matches) == 0:
        return []

    candidates = []
    similar_items = sorted(
        enumerate(similarity_scores[matches[0]]), key=lambda item: item[1], reverse=True
    )[1:top_n + 1]
    for idx, score in similar_items:
        title = pt.index[idx]
        row = books[books['Book-Title'] == title].drop_duplicates('Book-Title')
        if row.empty:
            continue
        candidates.append({
            'title': title,
            'author': row['Book-Author'].values[0],
            'image': row['Image-URL-M'].values[0] if 'Image-URL-M' in row.columns else '',
            'isbn': row['ISBN'].values[0] if 'ISBN' in row.columns else '',
            'similarity_score': float(score),
        })
    return candidates


def diversify_recommendations(items, top_n=8, max_per_author=2):
    """Keep the ranked list useful while preventing one author dominating it."""
    selected = []
    deferred = []
    author_counts = {}
    for item in items:
        author = str(item.get('author') or 'Unknown').casefold()
        if author_counts.get(author, 0) < max_per_author:
            selected.append(item)
            author_counts[author] = author_counts.get(author, 0) + 1
        else:
            deferred.append(item)
        if len(selected) >= top_n:
            return selected
    return (selected + deferred)[:top_n]


def get_ensemble_recommendations(book_title, user_id, top_n=8):
    """Blend all available recommenders into one normalized ranking."""
    seen_titles = get_seen_titles(user_id)
    seen_keys = {title.casefold() for title in seen_titles}
    admin_book = get_admin_book(title=book_title)
    source_top_n = min(15, max(10, top_n * 2))
    sources = [
        ('content', get_content_based_recommendations(book_title, top_n=source_top_n)),
        ('collaborative', get_collaborative_recommendations(book_title, top_n=source_top_n)),
        ('bookshub', get_bookshub_hybrid(book_title, user_id=user_id, top_n=source_top_n)),
        ('sequential', get_sasrec_recommendations(user_id, top_n=source_top_n)),
    ]
    if admin_book:
        sources.insert(0, ('admin_catalog', [{
            'title': admin_book.title,
            'author': admin_book.author,
            'image': admin_book.image_url,
            'isbn': admin_book.isbn,
            'similarity_score': 1.0,
            'explanation': 'Recently added to the catalog',
        }]))
    merged = {}
    active_sources = [source for source in sources if source[1]]
    if not active_sources:
        return get_popular_fallback(top_n=top_n, exclude_titles=seen_titles)

    weight = 1.0 / len(active_sources)
    for source_name, candidates in active_sources:
        scores = [float(item.get('similarity_score', 0.0)) for item in candidates]
        minimum, maximum = min(scores), max(scores)
        spread = maximum - minimum
        for item in candidates:
            title = str(item['title'])
            if title.casefold() in seen_keys:
                continue
            result = merged.setdefault(title, {
                'title': title,
                'author': item.get('author', 'Unknown'),
                'image': item.get('image', ''),
                'isbn': item.get('isbn', ''),
                'ensemble_score': 0.0,
                'sources': set(),
            })
            score = float(item.get('similarity_score', 0.0))
            result['ensemble_score'] += weight * ((score - minimum) / spread if spread else 1.0)
            result['sources'].add(source_name)

    ranked = sorted(merged.values(), key=lambda item: item['ensemble_score'], reverse=True)
    if len(ranked) < top_n:
        existing_titles = {item['title'] for item in ranked}
        ranked.extend(get_popular_fallback(
            top_n=top_n - len(ranked),
            exclude_titles=seen_titles | existing_titles,
        ))
    for item in ranked:
        score = float(item.pop('ensemble_score', item.get('similarity_score', 0.5)) or 0)
        item['similarity_score'] = round(score if np.isfinite(score) else 0.0, 4)
        item['match_percentage'] = recommendation_match_percent(item['similarity_score'])
        item['predicted_rating'] = round(min(item['similarity_score'] * 5, 5.0), 2)
        sources = item.pop('sources', set())
        item['explanation'] = f"Combined {len(sources)} recommendation models" if sources else item.get('explanation', 'Popular with readers')
    return diversify_recommendations(ranked, top_n=top_n)


@app.route('/new-arrivals')
def new_arrivals():
    arrivals = Book.query.filter_by(is_admin_added=True).order_by(Book.created_at.desc()).all()
    return render_template('new_arrivals.html', books=arrivals, username=session.get('username'))


@app.route('/recommend')
@login_required
def recommend_ui():
    return render_template('recommend.html', data=[], username=session.get('username'))


@app.route('/recommend_books', methods=['POST'])
@login_required
def recommend():
    user_input = request.form.get('user_input', '').strip()
    if not user_input:
        return render_template('recommend.html',
                               error="Please enter a valid book title.",
                               username=session.get('username'))

    user_id = session['user_id']
    record_search_history(user_id, user_input)
    latest_rating = Rating.query.filter_by(user_id=user_id).order_by(Rating.created_at.desc()).first()
    cache_key = (
        user_id,
        user_input.casefold(),
        latest_rating.id if latest_rating else 0,
        latest_rating.created_at.isoformat() if latest_rating else ''
    )
    cached = recommendation_cache.get(cache_key)
    if cached and monotonic() - cached['created_at'] < recommendation_cache_ttl:
        data = cached['data']
    else:
        data = get_ensemble_recommendations(user_input, user_id, top_n=12)
        recommendation_cache[cache_key] = {'created_at': monotonic(), 'data': data}
        if len(recommendation_cache) > 128:
            oldest_key = min(recommendation_cache, key=lambda key: recommendation_cache[key]['created_at'])
            recommendation_cache.pop(oldest_key, None)
    if data:
        graph_titles = [item['title'][:22] + "..." if len(item['title']) > 22 else item['title'] for item in data]
        graph_scores = [item['similarity_score'] for item in data]
        return render_template(
            'recommend.html', data=data, graph_titles=graph_titles, graph_scores=graph_scores,
            search_query=user_input,
            algorithm_used="Automatic Ensemble (Content + Collaborative + Bookshub + SASRec)",
            username=session.get('username')
        )

    return render_template(
        'recommend.html',
        error=f"'{user_input}' not found in the recommendation models.",
        search_query=user_input,
        username=session.get('username')
    )

#--------------------SASRec recommend function---------------------------------#
def get_sasrec_recommendations(user_id, top_n=8):
    """Sequential recommendation based on user's rating history"""
    if sasrec_model is None or sasrec_isbn_to_id is None:
        return []

    ratings = Rating.query.filter_by(user_id=user_id).order_by(Rating.created_at.asc()).all()
    if len(ratings) < 1:
        return []

    seq_ids = []
    for r in ratings:
        isbn = str(r.isbn).strip()
        # try multiple formats
        if isbn in sasrec_isbn_to_id:
            seq_ids.append(sasrec_isbn_to_id[isbn])
        elif isbn.lstrip('0') in sasrec_isbn_to_id:
            seq_ids.append(sasrec_isbn_to_id[isbn.lstrip('0')])
        else:
            # fallback: match by title
            for mapped_isbn, item_id in sasrec_isbn_to_id.items():
                title = sasrec_id_to_title.get(item_id, '')
                if title and title.lower() == str(r.book_title).lower():
                    seq_ids.append(item_id)
                    break

    if not seq_ids:
        # Last fallback: use any rated titles present in id_to_title
        title_to_id = {v.lower(): k for k, v in sasrec_id_to_title.items()}
        for r in ratings:
            t = str(r.book_title).lower()
            if t in title_to_id:
                seq_ids.append(title_to_id[t])

    if not seq_ids:
        return []

    max_len = 20
    if len(seq_ids) > max_len:
        seq_ids = seq_ids[-max_len:]
    else:
        seq_ids = [0] * (max_len - len(seq_ids)) + seq_ids

    input_tensor = torch.tensor([seq_ids], dtype=torch.long)
    exclude = [i for i in seq_ids if i > 0]

    try:
        top_ids, top_scores = sasrec_model.recommend_next(
            input_tensor, top_k=top_n + 5, exclude_ids=exclude
        )
    except Exception as e:
        print(f"SASRec predict error: {e}")
        return []

    results = []
    for item_id, score in zip(top_ids, top_scores):
        if len(results) >= top_n:
            break
        item_id = int(item_id)
        title = sasrec_id_to_title.get(item_id)
        isbn = sasrec_id_to_isbn.get(item_id, '')
        if not title:
            continue

        image, author = '', 'Unknown'
        if books is not None:
            temp = books[books['Book-Title'] == title]
            if temp.empty and isbn:
                temp = books[books['ISBN'].astype(str) == str(isbn)]
            if not temp.empty:
                author = temp['Book-Author'].values[0] if 'Book-Author' in temp.columns else 'Unknown'
                image = temp['Image-URL-M'].values[0] if 'Image-URL-M' in temp.columns else ''

        results.append({
            'title': title,
            'author': author,
            'image': image,
            'isbn': isbn,
            'similarity_score': round(float(score), 4),
            'predicted_rating': round(min(max(float(score) / 10, 0), 5), 2)
        })
    return results

# ====================== SEARCH AUTOCOMPLETE API ======================
@app.route('/api/search_books')
@login_required
def search_books_api():
    q = request.args.get('q', '').strip().casefold()
    if len(q) < 2:
        return jsonify([])

    admin_matches = Book.query.filter(
        db.or_(Book.title.ilike(f'%{q}%'), Book.author.ilike(f'%{q}%'))
    ).order_by(Book.created_at.desc()).limit(12).all()
    titles = [book.title for book in admin_matches]

    # Search from full books dataset (much better coverage)
    if books is not None:
        try:
            all_titles = books['Book-Title'].dropna().drop_duplicates()
            titles.extend(str(title) for title in all_titles if q in str(title).casefold())
            return jsonify(list(dict.fromkeys(titles))[:12])
        except Exception:
            pass

    # Fallback to pt.index if books failed
    if pt is not None:
        matches = [title for title in pt.index if q in str(title).casefold()][:10]
        return jsonify(matches)

    return jsonify([])


@app.route('/api/search-suggestions')
def search_suggestions_api():
    query = request.args.get('q', '').strip().casefold()
    if len(query) < 2:
        return jsonify([])

    suggestions = []
    seen_titles = set()

    def add_suggestion(title, author, image):
        normalized_title = str(title or '').strip()
        if not normalized_title or normalized_title.casefold() in seen_titles:
            return
        seen_titles.add(normalized_title.casefold())
        suggestions.append({
            'title': normalized_title,
            'author': str(author or 'Unknown').strip(),
            'thumbnail': normalize_image_url(image),
        })

    for book in Book.query.filter(
        db.or_(Book.title.ilike(f'%{query}%'), Book.author.ilike(f'%{query}%'))
    ).order_by(Book.created_at.desc()).limit(6).all():
        add_suggestion(book.title, book.author, book.image_url)

    if books is not None and len(suggestions) < 6:
        try:
            title_matches = books['Book-Title'].astype(str).str.casefold().str.contains(
                query, na=False, regex=False
            )
            author_matches = books['Book-Author'].astype(str).str.casefold().str.contains(
                query, na=False, regex=False
            ) if 'Book-Author' in books.columns else False
            matches = books[title_matches | author_matches].drop_duplicates('Book-Title').head(6 - len(suggestions))
            for _, row in matches.iterrows():
                add_suggestion(row.get('Book-Title'), row.get('Book-Author'), row.get('Image-URL-M'))
        except (KeyError, TypeError, ValueError):
            pass
    return jsonify(suggestions[:6])


# ====================== BOOK DETAILS ======================
@app.route('/book_details')
@login_required
def book_details():
    book_title = unquote(request.args.get('title', ''))
    if not book_title:
        flash("No book selected.", "warning")
        return redirect(url_for('index'))

    admin_book = get_admin_book(title=book_title)
    if admin_book:
        catalog_book = book_to_catalog_row(admin_book)
        isbn = catalog_book['ISBN']
        already_rated = has_user_rated(session['user_id'], isbn)
        in_wishlist = is_in_wishlist(session['user_id'], isbn)
        add_recently_viewed(session['user_id'], isbn, admin_book.title, admin_book.author, admin_book.image_url)
        return render_template('book_details.html', book={
            'title': admin_book.title, 'author': admin_book.author, 'image': admin_book.image_url,
            'description': admin_book.description or 'No description available.', 'isbn': isbn,
            'already_rated': already_rated, 'in_wishlist': in_wishlist,
        }, username=session.get('username'))

    if books is None:
        flash("Books data not loaded.", "danger")
        return redirect(url_for('index'))
    temp_df = books[books['Book-Title'] == book_title].drop_duplicates('Book-Title')
    if temp_df.empty:
        flash("Book not found.", "danger")
        return redirect(url_for('index'))

    isbn = temp_df['ISBN'].values[0] if 'ISBN' in temp_df.columns else ''
    already_rated = has_user_rated(session['user_id'], isbn) if isbn else False
    in_wishlist = is_in_wishlist(session['user_id'], isbn) if isbn else False

    # Save to recently viewed
    add_recently_viewed(
        session['user_id'],
        isbn,
        temp_df['Book-Title'].values[0],
        temp_df['Book-Author'].values[0],
        temp_df['Image-URL-M'].values[0] if 'Image-URL-M' in temp_df.columns else ''
    )

    book_info = {
        'title': temp_df['Book-Title'].values[0],
        'author': temp_df['Book-Author'].values[0],
        'image': temp_df['Image-URL-M'].values[0],
        'description': 'Loading description...',
        'isbn': isbn,
        'already_rated': already_rated,
        'in_wishlist': in_wishlist
    }

    return render_template('book_details.html', book=book_info, username=session.get('username'))


@app.route('/api/book_description')
@login_required
def book_description_api():
    title = request.args.get('title', '').strip()
    author = request.args.get('author', '').strip()
    if not title:
        return jsonify({'description': 'No description available.'}), 400
    return jsonify({'description': get_book_description(title, author)})


@app.route('/rate_book', methods=['POST'])
@login_required
def rate_book():
    user_id = session.get('user_id')
    isbn = request.form.get('isbn', '').strip()
    rating = request.form.get('rating')
    book_title = request.form.get('book_title', '').strip()

    if not all([user_id, isbn, rating, book_title]):
        flash("Missing data.", "danger")
        return redirect(url_for('book_details', title=quote(book_title)))

    catalog_book = get_catalog_book(isbn=isbn, title=book_title)
    if catalog_book is None:
        flash("That book could not be verified in the catalog.", "danger")
        return redirect(url_for('index'))
    isbn = str(catalog_book.get('ISBN', isbn)).strip()
    book_title = str(catalog_book.get('Book-Title', book_title)).strip()

    try:
        rating = float(rating)
        if not np.isfinite(rating) or rating < 1 or rating > 5:
            raise ValueError
    except (TypeError, ValueError):
        flash("Rating must be between 1 and 5.", "danger")
        return redirect(url_for('book_details', title=quote(book_title)))

    if has_user_rated(user_id, isbn):
        flash("You have already rated this book.", "warning")
        return redirect(url_for('book_details', title=quote(book_title)))

    new_rating = Rating(user_id=user_id, isbn=isbn, book_title=book_title, rating=rating)
    db.session.add(new_rating)
    db.session.commit()

    flash("✅ Rating submitted successfully!", "success")
    return redirect(url_for('book_details', title=quote(book_title)))


# ====================== WISHLIST ======================
@app.route('/add_wishlist', methods=['POST'])
@login_required
def add_wishlist():
    user_id = session['user_id']
    isbn = request.form.get('isbn', '').strip()
    book_title = request.form.get('book_title', '').strip()
    if not isbn or not book_title:
        flash("Invalid book data.", "danger")
        return redirect(request.referrer or url_for('index'))

    catalog_book = get_catalog_book(isbn=isbn, title=book_title)
    if catalog_book is None:
        flash("That book could not be verified in the catalog.", "danger")
        return redirect(url_for('index'))
    isbn = str(catalog_book.get('ISBN', isbn)).strip()
    book_title = str(catalog_book.get('Book-Title', book_title)).strip()
    author = str(catalog_book.get('Book-Author', 'Unknown'))
    image = normalize_image_url(catalog_book.get('Image-URL-M', ''))

    if is_in_wishlist(user_id, isbn):
        flash("Already in your wishlist.", "info")
        return redirect(request.referrer or url_for('index'))

    item = Wishlist(user_id=user_id, isbn=isbn, book_title=book_title, author=author, image=image)
    db.session.add(item)
    db.session.commit()
    flash("❤️ Added to Wishlist!", "success")
    return redirect(request.referrer or url_for('index'))


@app.route('/remove_wishlist', methods=['POST'])
@login_required
def remove_wishlist():
    user_id = session['user_id']
    isbn = request.form.get('isbn', '').strip()

    item = Wishlist.query.filter_by(user_id=user_id, isbn=isbn).first()
    if item:
        db.session.delete(item)
        db.session.commit()
        flash("Removed from Wishlist.", "info")
    return redirect(request.referrer or url_for('wishlist'))


@app.route('/wishlist')
@login_required
def wishlist():
    items = get_user_wishlist(session['user_id'])
    return render_template('wishlist.html', items=items, username=session.get('username'))


# ====================== MY RATINGS ======================
@app.route('/profile')
@login_required
def profile():
    user = User.query.get(session['user_id'])
    total_ratings = Rating.query.filter_by(user_id=session['user_id']).count()
    total_wishlist = Wishlist.query.filter_by(user_id=session['user_id']).count()
    recent_ratings = get_user_ratings(session['user_id'])[:5]

    return render_template(
        'profile.html',
        user=user,
        total_ratings=total_ratings,
        total_wishlist=total_wishlist,
        recent_ratings=recent_ratings,
        username=session.get('username')
    )

# ====================== MY_RATINGS ======================
@app.route('/my_ratings')
@login_required
def my_ratings():
    ratings = get_user_ratings(session['user_id'])
    return render_template('my_ratings.html', ratings=ratings, username=session.get('username'))
# ====================== ABOUT ======================
@app.route('/about')
def about_ui():
    return render_template('about.html', username=session.get('username'))


@app.route('/evaluation')
@login_required
def evaluation():
    model_status = {
        'Content TF-IDF': content_books is not None and tfidf_matrix is not None,
        'Collaborative TruncatedSVD': pt is not None and similarity_scores is not None,
        'Bookshub Hybrid': hub_books_df is not None and hub_tfidf_matrix is not None,
        'SASRec': sasrec_model is not None and sasrec_isbn_to_id is not None,
    }
    metrics = get_evaluation_metrics()
    metric_values = metrics.get('metrics', {}) if metrics else {}
    metric_rows = [
        {'name': 'Precision@8', 'value': metric_values.get('precision_at_8', 'Not evaluated'), 'note': 'Relevant recommendations in the top eight'},
        {'name': 'Recall@8', 'value': metric_values.get('recall_at_8', 'Not evaluated'), 'note': 'Held-out relevant items recovered in the top eight'},
        {'name': 'NDCG@8', 'value': metric_values.get('ndcg_at_8', 'Not evaluated'), 'note': 'Ranking quality with position-aware gains'},
    ]
    return render_template(
        'evaluation.html',
        model_status=model_status,
        metric_rows=metric_rows,
        catalog_size=len(books) if books is not None else 0,
        evaluated_at=metrics.get('evaluated_at') if metrics else None,
        evaluation_available=bool(metrics),
        username=session.get('username'),
    )


@app.post('/api/evaluation/trigger')
@admin_required
def trigger_evaluation():
    try:
        result = subprocess.run(
            [sys.executable, str(BASE_DIR / 'evaluate_models.py')],
            cwd=BASE_DIR,
            capture_output=True,
            text=True,
            timeout=600,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return jsonify({'error': 'Evaluation could not be completed in the allotted time.'}), 500
    if result.returncode != 0:
        return jsonify({'error': 'Evaluation failed.', 'details': result.stderr[-1000:]}), 500
    return jsonify({'message': 'Evaluation completed.', 'metrics': get_evaluation_metrics()})


# ====================== CREATE TABLES ======================
with app.app_context():
    db.create_all()
    inspector = db.inspect(db.engine)
    user_columns = {column['name'] for column in inspector.get_columns('user')}
    if 'is_admin' not in user_columns:
        db.session.execute(db.text('ALTER TABLE user ADD COLUMN is_admin BOOLEAN NOT NULL DEFAULT 0'))
        db.session.commit()
    configured_admin_email = os.environ.get('BRS_ADMIN_EMAIL', 'abi2003nas@gmail.com').strip().lower()
    if configured_admin_email:
        admin_user = User.query.filter(db.func.lower(User.email) == configured_admin_email).first()
        if admin_user and not admin_user.is_admin:
            admin_user.is_admin = True
            db.session.commit()
    print("✅ Database tables ready!")


if __name__ == '__main__':
    debug = os.environ.get('FLASK_DEBUG', '').casefold() == 'true'
    app.run(debug=debug, host=os.environ.get('FLASK_HOST', '127.0.0.1'), port=int(os.environ.get('PORT', 5000)))
    