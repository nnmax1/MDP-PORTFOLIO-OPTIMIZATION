from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import pandas as pd
import PortfolioMDPEnv 

# Load data from csv file 
prices_df = pd.read_csv("market_data.csv", index_col=0, parse_dates=True)
hedge_df = pd.read_csv("hedge_instrument.csv", index_col=0, parse_dates=True)

env = DummyVecEnv([lambda: PortfolioMDPEnv(prices_df, hedge_df)])

model = PPO(
    policy="MlpPolicy",
    env=env,
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    verbose=1
)

model.learn(total_timesteps=150_000)
model.save("portfolio_mdp_agent")