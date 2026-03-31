"""
lstm_model.py
带注意力机制的 LSTM 水质预测模型。

注意力机制让模型自动学习"过去哪些时间步对预测最重要"，
而不是只看最后一步。同时注意力权重本身可用于可解释性分析。

【改进C】hidden_size 64→96，输出层按比例加宽(96→48→6)，
使更大的特征空间（19维输入）有更充分的表示能力。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class WaterQualityLSTM(nn.Module):
    """
    LSTM + Attention 多步预测模型

    输入: (batch, lookback, input_size)   例如 (32, 18, 19)
    输出: (batch, forecast_horizon)       例如 (32, 6)
           + attention_weights (batch, lookback)  例如 (32, 18)
    """

    def __init__(
        self,
        input_size: int = 19,       # 【改进B/C】13→19（增加6个滚动特征）
        hidden_size: int = 96,      # 【改进C】64→96
        num_layers: int = 2,
        dropout: float = 0.2,
        forecast_horizon: int = 6,
    ):
        super().__init__()
        self.hidden_size = hidden_size

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )

        # 注意力层：学习每个时间步的重要性权重
        self.attn_fc = nn.Linear(hidden_size, 1)

        # 输出层：从注意力加权的隐状态映射到预测值
        # 【改进C】中间层宽度按 hidden_size 比例设置，过渡更平滑
        mid_size = hidden_size // 2   # 96→48
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, mid_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(mid_size, forecast_horizon),
        )

    def forward(self, x, return_attention=False):
        # x: (batch, seq_len, input_size)
        lstm_out, _ = self.lstm(x)  # (batch, seq_len, hidden_size)

        # 计算注意力权重
        attn_scores = self.attn_fc(lstm_out).squeeze(-1)  # (batch, seq_len)
        attn_weights = F.softmax(attn_scores, dim=1)      # (batch, seq_len)

        # 用注意力权重对所有时间步的隐状态做加权求和
        # attn_weights: (batch, 1, seq_len) × lstm_out: (batch, seq_len, hidden)
        context = torch.bmm(
            attn_weights.unsqueeze(1), lstm_out
        ).squeeze(1)  # (batch, hidden_size)

        output = self.fc(context)  # (batch, forecast_horizon)

        if return_attention:
            return output, attn_weights
        return output
