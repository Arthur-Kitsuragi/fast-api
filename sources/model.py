import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

class MyModel(nn.Module):
    """
    Two-layer BiLSTM model for text classification.

    Args:
    max_tokens (int): Dictionary size (number of unique tokens).
    embedding_dim (int): Dimensionality of the embedding layer.
    lstm_units1 (int): Number of hidden units in the first LSTM.
    lstm_units2 (int): Number of hidden units in the second LSTM.
    dense_units (int): Size of the dense layer before the output.
    dropout (float): Dropout after the LSTM.
    dropout1 (float): Dropout after the dense layer.
    num_classes (int): Number of classes to classify.
    """
    def __init__(self, max_tokens, embedding_dim, lstm_units1, lstm_units2, dense_units, dropout, dropout1,
                 num_classes):
        super(MyModel, self).__init__()

        self.embedding = nn.Embedding(num_embeddings=max_tokens, embedding_dim=embedding_dim, padding_idx=0)

        self.lstm1 = nn.LSTM(input_size=embedding_dim, hidden_size=lstm_units1, batch_first=True, bidirectional=True)
        self.lstm2 = nn.LSTM(input_size=2 * lstm_units1, hidden_size=lstm_units2, batch_first=True, bidirectional=True)

        self.dropout = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout1)

        self.fc = nn.Linear(2 * lstm_units2, dense_units)
        self.out = nn.Linear(dense_units, num_classes)

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of the model.

        Args:
        Tensor: Input tensor with token indices (batch_size, seq_len).

        Returns:
        Tensor: Output logits of the model (batch_size, num_classes).
        """
        x = x.long()
        x = self.embedding(x)

        x, _ = self.lstm1(x)
        x, _ = self.lstm2(x)

        x = x[:, -1, :]

        x = self.dropout(x)
        x = F.relu(self.fc(x))
        x = self.dropout1(x)
        x = self.out(x)

        return x