import torch
from torch import nn
from torch.nn import functional as F


class AttentionHead(nn.Module):

    def __init__(
        self,
        dim_embedding: int,
        dim_head: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.dim_head = dim_head
        self.key = nn.Linear(dim_embedding, dim_head)
        self.query = nn.Linear(dim_embedding, dim_head)
        self.value = nn.Linear(dim_embedding, dim_head)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.tensor,
    ) -> torch.tensor:
        K = self.key(x)
        Q = self.query(x)
        V = self.value(x)
        scores = Q @ K.transpose(-2, -1) / self.dim_head**0.5
        scores_autoregressive_mask = scores.masked_fill(torch.tril(scores) == 0, float('-inf'))
        attention_weights = F.softmax(scores_autoregressive_mask, dim=-1)
        attention_weights = self.dropout(attention_weights)
        out = attention_weights @ V
        return out


class MultiHeadedAttention(nn.Module):

    def __init__(
        self,
        dim_embedding: int,
        dim_head: int,
        num_heads: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.heads = nn.ModuleList([AttentionHead(dim_embedding, dim_head, dropout) for i in range(num_heads)])
        self.linear = nn.Linear(dim_head * num_heads, dim_embedding)

    def forward(self, x: torch.tensor) -> torch.tensor:
        x = torch.cat([head(x) for head in self.heads], dim=-1)
        x = self.linear(x)
        return x


class TransformerBlock(nn.Module):

    def __init__(
        self,
        dim_embedding: int,
        dim_head: int,
        num_heads: int,
        dim_mlp: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.layernorm_1 = nn.LayerNorm(dim_embedding)
        self.attention = MultiHeadedAttention(dim_embedding, dim_head, num_heads, dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.layernorm_2 = nn.LayerNorm(dim_embedding)
        self.mlp = nn.Sequential(
            nn.Linear(dim_embedding, dim_mlp),
            nn.GELU(),
            nn.Linear(dim_mlp, dim_embedding),
        )
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        x = x + self.dropout1(self.attention(self.layernorm_1(x)))
        x = x + self.dropout2(self.mlp(self.layernorm_2(x)))
        return x


WINDOW_SIZE = 100
DIM_EMBEDDING = 64
DIM_HEAD = 32
NUM_HEADS = 8
DIM_MLP = 128
DROPOUT = 0.3
NUM_BLOCKS = 3


class SimpleGPT(nn.Module):

    def __init__(
        self,
        vocab_size: int,
    ):
        super().__init__()
        self.WINDOW_SIZE
        self.DIM_EMBEDDING
        self.DIM_HEAD
        self.NUM_HEADS
        self.DIM_MLP
        self.DROPOUT
        self.NUM_BLOCKS
        self.token_emb = nn.Embedding(vocab_size, DIM_EMBEDDING)
        self.position_emb = nn.Embedding(WINDOW_SIZE, DIM_EMBEDDING)
        self.transformer_blocks = nn.Sequential(
            *[TransformerBlock(DIM_EMBEDDING, DIM_HEAD, NUM_HEADS, DIM_MLP, DROPOUT) for i in range(NUM_BLOCKS)]
        )
        self.layernorm = nn.LayerNorm(DIM_EMBEDDING)
        self.classification_head = nn.Linear(DIM_EMBEDDING, vocab_size)

    def forward(
        self,
        x: torch.tensor,  # Tensor of token vocab indices of shape (B, T) = (batch size, sequence size)
        y: torch.tensor = None,
    ):
        B, T = x.shape
        token_embedding = self.token_emb(x)
        position_embedding = self.position_emb(torch.arange(T))
        x = token_embedding + position_embedding
        x = self.transformer_blocks(x)
        x = self.layernorm(x)
        logits = self.classification_head(x)
        if y is None:
            return logits
        else:
            B, T, C = logits.shape
            loss = F.cross_entropy(logits.view(B*T, C), y.view(B*T))
            return loss

    def generate(
        self,
        x: torch.tensor,
        new_tokens: int,
    ):
        for _ in range(new_tokens):
            x_lookback = x[:, -WINDOW_SIZE:]
            logits = self.forward(x_lookback)[:, -1, :] # Take final item per batch
            probability = F.softmax(logits, dim=-1)
            out = torch.multinomial(probability, num_samples=1)
            x = torch.cat((x, out), dim=1)
        return x
