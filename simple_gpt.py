import torch
from torch import nn
from torch.nn import functional as F


# Global variables
EMBEDDING_DIM = 32
WINDOW_SIZE = 200


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
        x: torch.tensor, # Batch of token indices
        y: torch.tensor = None,
    ):
        B, T = x.shape  # batch size, sequence length
        token_embedding = self.token_emb(x)
        position_embedding = self.position_emb(torch.arange(T))
        x = token_embedding + position_embedding
        x = self.final_linear(x)
        return x
        # TODO: compute loss if y is not None


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
