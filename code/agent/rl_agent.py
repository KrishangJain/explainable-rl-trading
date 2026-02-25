import torch
import torch.nn as nn


class RLAgent:
    """
    Deep RL policy (Q-network style).
    """

    def __init__(self, state_dim: int, action_dim: int):
        self.model = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
        )

    def predict(self, state_tensor: torch.Tensor):
        return self.model(state_tensor)