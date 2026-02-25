import numpy as np
import torch


class SHAPExplainer:
    """
    Lightweight SHAP-style explainer using feature perturbation
    on the model output ONLY (no recursion).
    """

    def __init__(self, model):
        self.model = model

    def explain(self, state: np.ndarray, action_idx: int):
        state = state.astype(float)

        # Base prediction
        with torch.no_grad():
            base_q = self.model(
                torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            )[0, action_idx].item()

        shap_values = {}
        feature_names = ["price", "cash", "shares_held"]

        for i, name in enumerate(feature_names):
            perturbed = state.copy()
            perturbed[i] *= 0.9  # small perturbation

            with torch.no_grad():
                pert_q = self.model(
                    torch.tensor(perturbed, dtype=torch.float32).unsqueeze(0)
                )[0, action_idx].item()

            shap_values[name] = base_q - pert_q

        return shap_values