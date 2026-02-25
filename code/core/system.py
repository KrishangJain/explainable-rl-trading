import numpy as np
import torch

from agent.rl_agent import RLAgent
from env.trading_env import TradingEnvironment
from explainability.gradient_explainer import GradientExplainer
from explainability.shap_explainer import SHAPExplainer


_agent = RLAgent(state_dim=3, action_dim=3)
_env = TradingEnvironment()

_grad_explainer = GradientExplainer(_agent.model)
_shap_explainer = SHAPExplainer(_agent.model)


def explain_decision(state: np.ndarray):
    """
    Full decision + explanation pipeline.
    """

    state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0)

    q_values = _agent.predict(state_t)
    action_idx = torch.argmax(q_values, dim=1).item()
    action = _env.ACTION_MAP[action_idx]

    confidence = float(
        (q_values[0, action_idx] - q_values.mean()).detach().numpy()
    )

    gradients = _grad_explainer.explain(state, action_idx)
    shap_vals = _shap_explainer.explain(state, action_idx)

    explanation = (
        f"The agent chose {action} because changes in price, cash, "
        f"and shares held significantly affected the predicted return."
    )

    return {
        "action": action,
        "confidence": round(confidence, 4),
        "gradients": gradients,
        "shap": shap_vals,
        "text": explanation,
    }


def what_if_analysis(state: np.ndarray):
    base_action = explain_decision(state)["action"]

    deltas = {
        "price": np.array([5, 0, 0]),
        "cash": np.array([0, 200, 0]),
        "shares_held": np.array([0, 0, 1]),
    }

    results = {}
    for k, d in deltas.items():
        new_action = explain_decision(state + d)["action"]
        results[k] = (new_action == base_action)

    return results