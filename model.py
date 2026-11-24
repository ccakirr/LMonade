from embedding import LMonadeEmbedding
from torch import nn


class LMonadeModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, context_length):
        super().__init__()
        self.embedding = LMonadeEmbedding(
            vocab_size=vocab_size,
            context_length=context_length,
            embedding_dim=embedding_dim
        )

    def forward(self, x):
        return self.embedding(x)
