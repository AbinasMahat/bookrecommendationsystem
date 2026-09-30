"""
SASRec Training for Book Library System
Uses real ratings from SQLite + book mappings
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pickle
import sqlite3
import os
from collections import defaultdict
from datetime import datetime

torch.manual_seed(42)
np.random.seed(42)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")


# ====================== SASRec MODEL ======================
class SASRec(nn.Module):
    def __init__(self, num_items, max_len=20, hidden_units=64, num_heads=2, num_blocks=1, dropout_rate=0.2):
        super(SASRec, self).__init__()
        self.num_items = num_items
        self.max_len = max_len

        self.item_emb = nn.Embedding(num_items + 1, hidden_units, padding_idx=0)
        self.pos_emb = nn.Embedding(max_len, hidden_units)
        self.emb_dropout = nn.Dropout(dropout_rate)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_units,
            nhead=num_heads,
            dim_feedforward=hidden_units * 4,
            dropout=dropout_rate,
            activation="relu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_blocks)
        self.layer_norm = nn.LayerNorm(hidden_units)

    def generate_causal_mask(self, seq_len, device):
        mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1).bool()
        return mask

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

            top_scores, top_indices = torch.topk(scores, k=top_k, dim=-1)
            return (top_indices + 1).cpu().numpy()[0], top_scores.cpu().numpy()[0]


# ====================== DATASET ======================
class SequentialBookDataset(Dataset):
    def __init__(self, sequences, max_len=20):
        self.samples = []
        self.max_len = max_len

        for seq in sequences:
            if len(seq) < 2:
                continue
            for i in range(1, len(seq)):
                input_subseq = seq[:i]
                target_item = seq[i]

                if len(input_subseq) > max_len:
                    input_subseq = input_subseq[-max_len:]
                else:
                    input_subseq = [0] * (max_len - len(input_subseq)) + input_subseq

                self.samples.append((input_subseq, target_item))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        seq, target = self.samples[idx]
        return torch.tensor(seq, dtype=torch.long), torch.tensor(target, dtype=torch.long)


# ====================== LOAD DATA ======================
print("=" * 60)
print("SASRec Training Started")
print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 60)

# Load books mapping
books = None
try:
    with open('books.pkl', 'rb') as f:
        books = pickle.load(f)
    print(f"Loaded books.pkl: {len(books)} rows")
except Exception as e:
    print(f"Could not load books.pkl: {e}")

# Build title <-> id mapping from ratings + books
db_paths = ['instance/brs_database.db', 'brs_database.db']
db_path = None
for p in db_paths:
    if os.path.exists(p):
        db_path = p
        break

user_sequences = defaultdict(list)
isbn_to_id = {}
id_to_isbn = {}
id_to_title = {}
next_id = 1

if db_path:
    print(f"Loading ratings from: {db_path}")
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    try:
        rows = c.execute(
            "SELECT user_id, isbn, book_title, rating, created_at FROM rating ORDER BY user_id, created_at"
        ).fetchall()
        print(f"Found {len(rows)} ratings")

        for user_id, isbn, title, rating, created_at in rows:
            if rating < 3:  # only positive interactions
                continue
            isbn = str(isbn)
            if isbn not in isbn_to_id:
                isbn_to_id[isbn] = next_id
                id_to_isbn[next_id] = isbn
                id_to_title[next_id] = title
                next_id += 1
            user_sequences[user_id].append(isbn_to_id[isbn])
    except Exception as e:
        print(f"Error reading ratings: {e}")
    conn.close()
else:
    print("WARNING: No database found. Using sample data.")

# If too few real sequences, add synthetic from popular books
sequences = [seq for seq in user_sequences.values() if len(seq) >= 2]
print(f"User sequences with >=2 items: {len(sequences)}")

if len(sequences) < 10:
    print("Not enough real sequences. Augmenting with synthetic data from books...")
    if books is not None and 'ISBN' in books.columns:
        sample_isbns = books['ISBN'].dropna().unique()[:200]
        for i, isbn in enumerate(sample_isbns):
            isbn = str(isbn)
            if isbn not in isbn_to_id:
                isbn_to_id[isbn] = next_id
                id_to_isbn[next_id] = isbn
                title_row = books[books['ISBN'] == isbn]
                id_to_title[next_id] = title_row['Book-Title'].values[0] if not title_row.empty else f"Book_{next_id}"
                next_id += 1

        # Create synthetic sequences
        all_ids = list(id_to_isbn.keys())
        np.random.seed(42)
        for _ in range(50):
            length = np.random.randint(3, 8)
            seq = list(np.random.choice(all_ids, size=length, replace=False))
            sequences.append(seq)

NUM_ITEMS = max(id_to_isbn.keys()) if id_to_isbn else 100
print(f"Total unique items: {NUM_ITEMS}")
print(f"Training sequences: {len(sequences)}")

if NUM_ITEMS < 5 or len(sequences) < 2:
    print("ERROR: Not enough data to train SASRec.")
    exit(1)

# ====================== TRAIN ======================
MAX_LEN = 20
HIDDEN = 64
BATCH_SIZE = 32
EPOCHS = 8

dataset = SequentialBookDataset(sequences, max_len=MAX_LEN)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

model = SASRec(
    num_items=NUM_ITEMS,
    max_len=MAX_LEN,
    hidden_units=HIDDEN,
    num_heads=2,
    num_blocks=2,
    dropout_rate=0.2,
).to(DEVICE)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

print(f"\nTraining SASRec ({EPOCHS} epochs)...")
model.train()
for epoch in range(1, EPOCHS + 1):
    total_loss = 0.0
    for seq_inputs, target_items in dataloader:
        seq_inputs = seq_inputs.to(DEVICE)
        target_items = target_items.to(DEVICE)

        optimizer.zero_grad()
        out = model(seq_inputs)
        logits = torch.matmul(out[:, -1, :], model.item_emb.weight[1:].T)
        loss = criterion(logits, target_items - 1)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    avg_loss = total_loss / max(len(dataloader), 1)
    print(f"  Epoch {epoch}/{EPOCHS} | Loss: {avg_loss:.4f}")

# ====================== SAVE ======================
print("\nSaving model and mappings...")

torch.save({
    'model_state': model.state_dict(),
    'num_items': NUM_ITEMS,
    'max_len': MAX_LEN,
    'hidden_units': HIDDEN,
}, 'sasrec_model.pt')

pickle.dump(isbn_to_id, open('sasrec_isbn_to_id.pkl', 'wb'))
pickle.dump(id_to_isbn, open('sasrec_id_to_isbn.pkl', 'wb'))
pickle.dump(id_to_title, open('sasrec_id_to_title.pkl', 'wb'))

print("=" * 60)
print("SASRec TRAINING COMPLETED!")
print("=" * 60)
print("Saved files:")
print("  - sasrec_model.pt")
print("  - sasrec_isbn_to_id.pkl")
print("  - sasrec_id_to_isbn.pkl")
print("  - sasrec_id_to_title.pkl")