import torch
from torch import nn
from torch.nn import functional as F


# Global variables
EMBEDDING_DIM = 32
WINDOW_SIZE = 20


class AttentionHead(nn.Module):

    def __init__(
        self,
        dim_embedding: int,
        dim_head: int,
    ) -> None:
        super().__init__()
        self.dim_embedding = dim_embedding
        self.dim_head = dim_head
        self.key = nn.Linear(dim_embedding, dim_head)
        self.query = nn.Linear(dim_embedding, dim_head)
        self.value = nn.Linear(dim_embedding, dim_head)

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
        out = attention_weights @ V
        return out


class MultiHeadedAttention(nn.Module):

    def __init__(
        self,
        dim_embedding: int,
        dim_head: int,
        num_heads: int,
    ) -> None:
        super().__init__()
        self.dim_embedding = dim_embedding
        self.dim_head = dim_head
        self.num_heads = num_heads
        self.heads = nn.ModuleList([AttentionHead(dim_embedding, dim_head) for i in range(num_heads)])
        self.linear = nn.Linear(dim_head * num_heads, dim_embedding)

    def forward(self, x: torch.tensor) -> torch.tensor:
        x = torch.cat([head(x) for head in self.heads], dim=-1)
        print(f"x.shape = {x.shape}")
        x = self.linear(x)
        return x


class SimpleGPT(nn.Module):

    def __init__(
        self,
        vocab_size: int,
    ):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, EMBEDDING_DIM)
        self.position_emb = nn.Embedding(WINDOW_SIZE, EMBEDDING_DIM)
        self.final_linear = nn.Linear(EMBEDDING_DIM, vocab_size)

    def forward(
        self,
        x: torch.tensor,  # Tensor of token indices of shape (B, T)
        y: torch.tensor = None,
    ): # TODO: add type hint
        B, T = x.shape  # batch size, sequence length
        token_embedding = self.token_emb(x)
        position_embedding = self.position_emb(torch.arange(T))
        x = token_embedding + position_embedding
        logits = self.final_linear(x)
        if y is None:
            return logits
        else:
            B, T, C = logits.shape  # batch size, sequence length, number of classes
            loss = F.cross_entropy(logits.view(B*T, C), y.view(B*T))
            return loss

    def generate(
        self,
        x: torch.tensor,
        new_tokens: int,
    ):
        for _ in range(new_tokens):
            x_lookback = x[:, -WINDOW_SIZE:]
            logits = self.forward(x_lookback)[:, -1, :]
            probability = F.softmax(logits, dim=-1)
            out = torch.multinomial(probability, num_samples=1)
            x = torch.cat((x, out), dim=1)
        return x
