class TradingEnvironment:
    """
    One-step trading environment with actual trading logic.
    """

    ACTION_MAP = {
        0: "HOLD",
        1: "BUY",
        2: "SELL",
    }

    def step(self, state, action):
        price, cash, shares = state

        reward = 0.0

        if action == 1 and cash >= price:  # BUY
            cash -= price
            shares += 1
            reward = 1.0

        elif action == 2 and shares > 0:  # SELL
            cash += price
            shares -= 1
            reward = 1.0

        else:  # HOLD or invalid
            reward = -0.1

        next_state = [price, cash, shares]
        return next_state, reward