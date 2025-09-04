import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from transformers import AutoModel
from typing import Optional

class MyModel(nn.Module):
    """
     Custom text classification model combining:
       - a frozen BERT encoder,
       - a bidirectional LSTM,
       - a Transformer encoder,
       - and fully connected layers for classification.

     Args:
         bert_model_name (str): Name of the pretrained BERT model.
         lstm_units (int): Number of hidden units in the LSTM layer.
         dense_units (int): Number of hidden units in the dense (feedforward) layer.
         num_classes (int): Number of output classes for classification.
         nhead (int): Number of attention heads in the Transformer encoder.
         num_layers (int): Number of Transformer encoder layers.
         dropout (float): Dropout rate applied after the dense layer.

     Forward inputs:
         input_ids (torch.Tensor): Token IDs of shape (batch_size, seq_len).
         attention_mask (Optional[torch.Tensor]): Attention mask of shape (batch_size, seq_len).

     Forward outputs:
         torch.Tensor: Logits of shape (batch_size, num_classes).
     """
    def __init__(self, bert_model_name="bert-base-uncased", lstm_units=128,
                 dense_units=256, num_classes=20, nhead=4, num_layers=1, dropout=0.5):
        super(MyModel, self).__init__()

        self.bert = AutoModel.from_pretrained(bert_model_name)
        for param in self.bert.parameters():
            param.requires_grad = False

        bert_hidden_size = self.bert.config.hidden_size

        self.lstm = nn.LSTM(input_size=bert_hidden_size,
                            hidden_size=lstm_units,
                            batch_first=True,
                            bidirectional=True)

        encoder_layer = nn.TransformerEncoderLayer(d_model=2 * lstm_units,
                                                   nhead=nhead,
                                                   dim_feedforward=dense_units,
                                                   dropout=dropout,
                                                   batch_first=True)

        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.dropout = nn.Dropout(dropout)

        self.fc = nn.Linear(2 * lstm_units, dense_units)
        self.out = nn.Linear(dense_units, num_classes)

    def forward(self,
                input_ids: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass of the model.

        Args:
            input_ids (torch.Tensor): Input token IDs, shape (batch_size, seq_len).
            attention_mask (Optional[torch.Tensor]): Attention mask, shape (batch_size, seq_len).

        Returns:
            torch.Tensor: Logits of shape (batch_size, num_classes).
        """
        with torch.no_grad():
            bert_outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
            x = bert_outputs.last_hidden_state

        x, _ = self.lstm(x)
        x = self.transformer(x)

        x = x.mean(dim=1)

        x = self.dropout(F.relu(self.fc(x)))
        x = self.out(x)

        return x