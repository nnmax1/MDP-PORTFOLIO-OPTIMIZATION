from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.trading.client import TradingClient

import PortfolioMDPEnv 
import datetime 
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

# Initialize Alpaca clients
api_key = "YOUR_API_KEY"
api_secret = "YOUR_API_SECRET"
trading_client = TradingClient(api_key, api_secret)
data_client = StockHistoricalDataClient(api_key, api_secret)

# Load historical data for MDP environment
symbols = ["AAPL", "GOOGL", "MSFT", "AMZN"]
start_date = datetime(2020, 1, 1)
end_date = datetime(2026, 10, 5)
request = StockBarsRequest(symbol_or_symbols=symbols, timeframe=TimeFrame.Day, start=start_date, end=end_date)
bars = data_client.get_stock_bars(request)
prices_df = bars.df.unstack(level=0)

# Load hedging instrument data (e.g. VIX futures)
hedge_request = StockBarsRequest(symbol_or_symbols=["VIX"], timeframe=TimeFrame.Day, start=start_date, end=end_date)
hedge_bars = data_client.get_stock_bars(hedge_request)
hedge_df = hedge_bars.df.unstack(level=0)

# Initialize the MDP agent
model = PPO.load("portfolio_mdp_agent")
env = DummyVecEnv([lambda: PortfolioMDPEnv(prices_df, hedge_df)])

# Execution scheduling layer
def execute_trades(model, env):
    obs, _ = env.reset()
    while True:
        action, _ = model.predict(obs)
        next_obs, reward, done, info = env.step(action)

        # Map MDP actions to Alpaca orders
        portfolio_value = env.get_attribute("portfolio_value")
        target_weights = action / action.sum()
        cash_allocation = target_weights[0] * portfolio_value

        for symbol in symbols:
            asset_weight = target_weights[symbols.index(symbol)]
            shares_to_trade = asset_weight * portfolio_value / env.get_attribute(f"{symbol}_close")

            if shares_to_trade > env.get_attribute(f"{symbol}_shares"):
                order = MarketOrderRequest(
                    symbol=symbol, 
                    qty=round(shares_to_trade),
                    side=OrderSide.BUY, 
                    time_in_force=TimeInForce.DAY
                )
                trading_client.submit_order(order)
            else:
                shares_to_sell = env.get_attribute(f"{symbol}_shares") - shares_to_trade
                order = MarketOrderRequest(
                    symbol=symbol, 
                    qty=round(shares_to_sell),
                    side=OrderSide.SELL,
                    time_in_force=TimeInForce.DAY
                )
                trading_client.submit_order(order)

        if done:
            break

        obs = next_obs

# Run the trading strategy
execute_trades(model, env)