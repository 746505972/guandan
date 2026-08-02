"""神经网络模型定义"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from Game.utils.actions import get_action_dim
from Game.constants import MODEL_DIR

# ============ MLP 版本 (actor.py / FrontendGame / backend / display) ============

class MLPActorNet(nn.Module):
    """MLP Actor 网络"""

    def __init__(self, state_dim=3049, action_dim=None, hidden_dim=512):
        super().__init__()
        if action_dim is None:
            action_dim = get_action_dim()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )

    def forward(self, x, mask=None):
        logits = self.net(x)
        if mask is not None:
            logits = logits + (mask - 1) * 1e9
        return F.softmax(logits, dim=-1)


class MLPCriticNet(nn.Module):
    """MLP Critic 网络"""

    def __init__(self, state_dim=3049, action_dim=1, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )

    def forward(self, x):
        return self.net(x)


# ============ Transformer 版本 (A3C.py) ============

class TransformerActorNet(nn.Module):
    """Transformer Actor 网络"""

    def __init__(self, state_dim=3049, action_dim=None, hidden_dim=512, num_heads=8, num_layers=1):
        super().__init__()
        if action_dim is None:
            action_dim = get_action_dim()
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        self.input_projection = nn.Linear(state_dim, hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=num_heads, dim_feedforward=hidden_dim * 4,
            dropout=0.1, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU(),
            nn.Linear(hidden_dim // 2, action_dim))

    def forward(self, x, mask=None, temperature=1.0):
        x = self.input_projection(x)
        x = x.unsqueeze(1)
        transformer_output = self.transformer_encoder(x)
        transformer_output = transformer_output.squeeze(1)
        logits = self.output_net(transformer_output)
        if mask is not None:
            if mask.dim() == 1:
                mask = mask.unsqueeze(0).expand_as(logits)
            logits = logits.masked_fill(~mask, -1e9)
        return F.softmax(logits / temperature, dim=-1)


class TransformerCriticNet(nn.Module):
    """Transformer Critic 网络"""

    def __init__(self, state_dim=3049, hidden_dim=512, num_heads=8, num_layers=1):
        super().__init__()
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        self.input_projection = nn.Linear(state_dim, hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=num_heads, dim_feedforward=hidden_dim * 4,
            dropout=0.1, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim // 4), nn.ReLU(),
            nn.Linear(hidden_dim // 4, 1))

    def forward(self, x):
        x = self.input_projection(x)
        x = x.unsqueeze(1)
        transformer_output = self.transformer_encoder(x)
        transformer_output = transformer_output.squeeze(1)
        return self.output_net(transformer_output)


# ============ 向后兼容别名 ============
ActorNet = MLPActorNet
CriticNet = MLPCriticNet


def load_actor_model(model_name, model_type='mlp'):
    """加载 Actor 模型"""
    if model_type == 'mlp':
        model = MLPActorNet()
    elif model_type == 'transformer':
        model = TransformerActorNet()
    else:
        raise ValueError(f"未知模型类型: {model_type}")
    path = MODEL_DIR / model_name
    model.load_state_dict(torch.load(path, map_location='cpu'))
    model.eval()
    return model

if __name__ == "__main__":
    model = load_actor_model('base.pth')
    print(model)