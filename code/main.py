import numpy as np
from core.system import explain_decision, what_if_analysis

state = np.array([100, 1000, 0])
print(explain_decision(state))
print(what_if_analysis(state))