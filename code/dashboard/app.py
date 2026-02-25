import os
import sys

# --- Fix import path ---
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# ----------------------

import streamlit as st
import numpy as np
import plotly.express as px

from core.system import explain_decision, what_if_analysis


st.set_page_config(page_title="Explainable RL Trading System", layout="centered")

st.title("\Explainable RL Trading System")

# ---------------- UI ----------------

price = st.slider("Price", 80, 120, 100)
cash = st.slider("Cash", 0, 2000, 1000)
shares = st.slider("Shares Held", 0, 10, 0)

state = np.array([price, cash, shares], dtype=float)

if st.button("Run Decision"):
    result = explain_decision(state)
    what_if = what_if_analysis(state)

    st.subheader("Decision")
    st.write(f"**Action:** {result['action']}")
    st.write(f"**Confidence:** {result['confidence']}")

    st.subheader("SHAP Feature Importance")
    shap_df = {
        "Feature": list(result["shap"].keys()),
        "Impact": list(result["shap"].values()),
    }
    fig = px.bar(shap_df, x="Feature", y="Impact")
    st.plotly_chart(fig)

    st.subheader("Explanation")
    st.write(result["text"])

    st.subheader("What-if Analysis")
    for k, v in what_if.items():
        st.write(f"- Changing {k} does **{'NOT ' if v else ''}change decision**")