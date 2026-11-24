from torch import nn

import torch


def sinusoidal_encoding(context_length, embedding_dim):
    pe = torch.zeros(context_length, embedding_dim)

    positions = torch.arange(context_length).unsqueeze(1)

    i = torch.arange(0, embedding_dim, 2)

    denominator = 10000 ** (i / embedding_dim)

    pe[:, 0::2] = torch.sin(positions / denominator)
    pe[:, 1::2] = torch.cos(positions / denominator)

    return pe


class LMonadeEmbedding(nn.Module):
    def __init__(self, vocab_size, context_length, embedding_dim):
        super().__init__()

        self.token_embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim
        )

        self.register_buffer(
            "position_encoding",
            sinusoidal_encoding(context_length, embedding_dim)
        )

    def forward(self, x):
        return self.token_embedding(x) + self.position_encoding[:x.shape[1]]
