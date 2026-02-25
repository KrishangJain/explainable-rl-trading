import torch
import numpy as np


class GradientExplainer:
    def __init__(self, model):
        self.model = model

    def explain(self, state: np.ndarray, action_idx: int):
        state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        state_t.requires_grad_(True)

        q_values = self.model(state_t)
        target = q_values[0, action_idx]

        self.model.zero_grad()
        target.backward()

        grads = state_t.grad[0].detach().numpy()

        return {
            "price": float(grads[0]),
            "cash": float(grads[1]),
            "shares_held": float(grads[2]),
        }