import gymnasium as gym # OpenAI reinforcement learning library
import numpy as np
from gymnasium import spaces


class PortfolioMDPEnv(gym.Env):
    def __init__(self, asset_prices, hedge_prices=None, initial_capital=1_000_000,
                 tx_cost=0.001, risk_free_rate=0.04/252, risk_aversion=2.0, lookback=20):
        super().__init__()
        self.asset_prices = asset_prices.values
        self.hedge_prices = hedge_prices.values if hedge_prices is not None else None
        self.initial_capital = initial_capital
        self.tx_cost = tx_cost
        self.rf = risk_free_rate
        self.risk_aversion = risk_aversion
        self.lookback = lookback

        self.n_assets = asset_prices.shape[1]
        self.n_hedge = 1 if self.hedge_prices is not None else 0
        self.action_dim = self.n_assets + self.n_hedge

        # state: lookback returns + current weights + cash ratio + progress
        self.state_dim = self.n_assets * self.lookback + self.action_dim + 2
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.state_dim,), dtype=np.float32)
        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(self.action_dim,), dtype=np.float32)

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = self.lookback
        self.portfolio_value = self.initial_capital
        self.weights = np.ones(self.action_dim) / self.action_dim
        self.cash = self.initial_capital
        self.state = self._get_obs()
        return self.state, {}

    def _get_obs(self):
        # Rolling log returns
        asset_ret = np.diff(np.log(self.asset_prices[self.current_step-self.lookback:self.current_step]), axis=0).flatten()
        hedge_ret = np.array([])
        if self.hedge_prices is not None:
            hedge_ret = np.diff(np.log(self.hedge_prices[self.current_step-self.lookback:self.current_step]), axis=0).flatten()
        
        returns = np.concatenate([asset_ret, hedge_ret])
        cash_ratio = self.cash / self.portfolio_value
        progress = self.current_step / len(self.asset_prices)
        
        return np.concatenate([returns, self.weights, [cash_ratio, progress]]).astype(np.float32)

    def step(self, action):
        # normalize to valid portfolio weights
        target_weights = action / action.sum()
        
        # Turnover and transaction costs
        turnover = np.sum(np.abs(target_weights - self.weights))
        tx_costs = turnover * self.portfolio_value * self.tx_cost
        
        # Market returns for this step (t -> t+1)
        asset_ret = np.diff(np.log(self.asset_prices[self.current_step:self.current_step+2]), axis=0).flatten()
        if self.hedge_prices is not None:
            hedge_ret = np.diff(np.log(self.hedge_prices[self.current_step:self.current_step+2]), axis=0).flatten()
            market_ret = np.concatenate([asset_ret, hedge_ret])
        else:
            market_ret = asset_ret
            
        # portfolio update
        gross_ret = np.dot(self.weights, market_ret)
        self.portfolio_value = self.portfolio_value * (1 + gross_ret) - tx_costs
        self.cash = self.portfolio_value * (1 - np.sum(target_weights))
        self.weights = target_weights
        
        # Reward: excess return over risk-free, penalized by variance & costs
        excess_ret = gross_ret - self.rf
        reward = excess_ret - 0.5 * self.risk_aversion * np.var(market_ret) - (tx_costs / self.initial_capital)
        
        self.current_step += 1
        done = self.current_step >= len(self.asset_prices) - 1
        obs = self._get_obs() if not done else self._get_obs()
        
        return obs, reward, done, False, {}