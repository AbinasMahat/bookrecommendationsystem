import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)


class SASRec(nn.Module):
    """Self-Attention Sequential Recommendation Model (PyTorch)"""

    def __init__(
        self,
        num_items: int,
        max_len: int = 20,
        hidden_units: int = 64,
        num_heads: int = 2,
        num_blocks: int = 1,
        dropout_rate: int = 0.2,
    ):
        super(SASRec, self).__init__()
        self.num_items = num_items
        self.max_len = max_len

        # Token index 0 is reserved for padding
        self.item_emb = nn.Embedding(num_items + 1, hidden_units, padding_idx=0)
        self.pos_emb = nn.Embedding(max_len, hidden_units)
        self.emb_dropout = nn.Dropout(dropout_rate)

        # Transformer Encoder Stack
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_units,
            nhead=num_heads,
            dim_feedforward=hidden_units * 4,
            dropout=dropout_rate,
            activation="relu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=num_blocks
        )
        self.layer_norm = nn.LayerNorm(hidden_units)

    def generate_causal_mask(self, seq_len: int, device: torch.device):
        """Prevents attention heads from attending to future items in the sequence."""
        mask = torch.triu(
            torch.ones(seq_len, seq_len, device=device), diagonal=1
        ).bool()
        return mask

    def forward(self, input_seqs: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len = input_seqs.shape
        device = input_seqs.device

        # Positional indices [0, 1, ..., seq_len - 1]
        positions = (
            torch.arange(seq_len, device=device)
            .unsqueeze(0)
            .expand(batch_size, -1)
        )

        # Embeddings + Positional encoding
        seq_emb = self.item_emb(input_seqs) + self.pos_emb(positions)
        seq_emb = self.emb_dropout(seq_emb)

        padding_mask = input_seqs == 0
        causal_mask = self.generate_causal_mask(seq_len, device)

        # Transformer representation
        out = self.transformer(
            seq_emb, mask=causal_mask, src_key_padding_mask=padding_mask
        )
        return self.layer_norm(out)

    def recommend_next_book(
        self, input_seq: torch.Tensor, top_k: int = 5
    ) -> torch.Tensor:
        """Predict top-K candidate books given a sequence of past read books."""
        self.eval()
        with torch.no_grad():
            seq_out = self.forward(input_seq)
            last_item_repr = seq_out[
                :, -1, :
            ]  # User's current reading vector representation

            # Calculate similarity scores against all book item embeddings
            all_book_embs = self.item_emb.weight[1:]  # Exclude padding index 0
            scores = torch.matmul(last_item_repr, all_book_embs.T)

            top_scores, top_indices = torch.topk(scores, k=top_k, dim=-1)
            return top_indices + 1  # Offset index by 1 to match item IDs


class SequentialBookDataset(Dataset):
    """Dataset wrapper for sequential user interactions."""

    def __init__(self, sequences, max_len=20):
        self.samples = []
        self.max_len = max_len

        for seq in sequences:
            if len(seq) < 2:
                continue
            # Generate sliding sequence inputs and target next items
            for i in range(1, len(seq)):
                input_subseq = seq[:i]
                target_item = seq[i]

                # Right-pad or truncate sequence
                if len(input_subseq) > max_len:
                    input_subseq = input_subseq[-max_len:]
                else:
                    input_subseq = [0] * (
                        max_len - len(input_subseq)
                    ) + input_subseq

                self.samples.append((input_subseq, target_item))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        seq, target = self.samples[idx]
        return torch.tensor(seq, dtype=torch.long), torch.tensor(
            target, dtype=torch.long
        )


# --- Runnable Execution ---
if __name__ == "__main__":
    # Hyperparameters
    NUM_BOOKS = 100  # Total unique books (IDs 1 to 100)
    MAX_SEQ_LEN = 10
    HIDDEN_DIM = 32
    BATCH_SIZE = 16
    EPOCHS = 5
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Synthetic user reading histories (sequences of book IDs)
    raw_user_histories = [
        [5, 12, 18, 24, 30],
        [1, 2, 3, 4, 5, 6, 7],
        [12, 18, 30, 45, 50],
        [8, 15, 22, 29, 36, 43],
        [3, 7, 12, 18, 24],
    ] * 20  # Duplicate for demo dataset size

    dataset = SequentialBookDataset(raw_user_histories, max_len=MAX_SEQ_LEN)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # Initialize model, loss, and optimizer
    model = SASRec(
        num_items=NUM_BOOKS,
        max_len=MAX_SEQ_LEN,
        hidden_units=HIDDEN_DIM,
        num_heads=2,
        num_blocks=1,
    ).to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.005)

    # Training Loop
    print(f"Training SASRec on {DEVICE}...")
    model.train()
    for epoch in range(1, EPOCHS + 1):
        total_loss = 0.0
        for seq_inputs, target_items in dataloader:
            seq_inputs, target_items = seq_inputs.to(DEVICE), target_items.to(
                DEVICE
            )

            optimizer.zero_grad()
            out = model(seq_inputs)
            # Use final sequence step prediction to map against all target items
            last_step_logits = torch.matmul(
                out[:, -1, :], model.item_emb.weight[1:].T
            )

            # Targets are 1-indexed (adjust by -1 for CrossEntropy Loss)
            loss = criterion(last_step_logits, target_items - 1)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(
            f"Epoch {epoch}/{EPOCHS} | Loss: {total_loss / len(dataloader):.4f}"
        )

    # Testing Inference
    print("\n--- Inference Recommendation Test ---")
    sample_user_history = [5, 12, 18, 24]  # User recently read these books
    padded_seq = [0] * (MAX_SEQ_LEN - len(sample_user_history)) + sample_user_history
    input_tensor = torch.tensor(
        [padded_seq], dtype=torch.long, device=DEVICE
    )

    top_recommendations = model.recommend_next_book(
        input_tensor, top_k=5
    )
    print(f"Past Read Books: {sample_user_history}")
    print(
        f"Top 5 Recommended Next Books (Item IDs): {top_recommendations.cpu().numpy()[0].tolist()}"
    )