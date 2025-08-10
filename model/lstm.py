import torch
import torch.nn as nn
from opacus.layers import DPLSTM # وارد کردن لایه سازگار با حریم خصوصی

class LSTMNet(nn.Module):
    def __init__(self, input_size, hidden_size=32, num_layers=1):
        super(LSTMNet, self).__init__()
        # استفاده از DPLSTM به جای nn.LSTM
        self.lstm = DPLSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out
