# MDP PORTFOLIO OPTIMIZATION

production-oriented architecture for an MDP-based (Markov Decision Process) portfolio optimization agent.

Since financial state spaces are continuous and high-dimensional, the MDP is implemented using Deep Reinforcement Learning (PPO) as the policy approximator, which is the industry standard for solving continuous MDPs in quantitative finance.



# MDP Definition 
S_t -> Rolling log-returns, volatility estimates, current portfolio weights, cash ratio, time-to-horizon, hedging instrument prices 

A_t -> Continuous vector of target portfolio weights + hedge ratio (normalized to sum ≤ 1)

P(s' || s,a) -> Market price dynamics + execution slippage. Handled by the environment stepping through historical/simulated data

gamma -> 0.95–0.99 (emphasizes long-term compounding over short-term noise)


# Execution Scheduling & Dynamic Hedging Architecture

The base MDP outputs target allocations. In production, you decouple allocation from execution using a hierarchical MDP:

Allocation MDP (Above): Runs daily/weekly. Outputs target weights.

Execution MDP: Runs intraday. State = remaining order size, spread, volume profile, volatility. Action = trade slice size or limit order price. Reward = implementation shortfall minimization. Use TWAP/VWAP baselines + RL adjustment.

Dynamic Hedging Layer:
Add delta/gamma exposure to the state vector.
If portfolio delta exceeds threshold, the action space includes futures/options contracts.

Reward penalizes unhedged directional exposure and hedging carry cost.
Use a separate PPO policy trained on options chain data + underlying returns.

# Production Realities & Edge Cases
Slippage & Market Impact: Linear transaction costs underestimate real impact. Replace tx_cost with a square-root impact model: cost = k * sqrt(|trade| / daily_volume).

Regime Shifts: Standard MDPs assume stationary transitions. Inject regime flags (HMM-inferred bull/bear/vol) into the state vector to let the policy condition on market structure.

Overfitting: Use walk-forward validation. Train on rolling 3-year windows, validate on 6-month out-of-sample. Never optimize on a single static dataset.

Constraint Handling: Hard constraints (leverage limits, sector caps) should be enforced via action masking or post-action projection, not just reward penalties.

This architecture gives you a mathematically sound MDP foundation that scales to real trading infrastructure. 


# Alpaca Trading Implementation
Alpaca Client Initialization: We initialize the trading and data clients using your API credentials. The data client will fetch historical price data for our MDP environment.

Historical Data Loading: We request stock bars for the target assets (AAPL, GOOGL, MSFT, AMZN) and a hedging instrument (VIX futures) using the Alpaca Data API. This data will be used to simulate the market environment for the MDP agent.

MDP Agent Loading: We load the pre-trained PPO agent model that we saved earlier. This model will make the allocation decisions based on the current market state.

Execution Scheduling Layer: The execute_trades function implements the execution scheduling logic:

We reset the environment to get the initial observation.
In a loop, we predict the next action using the MDP agent.
We map the MDP action (target weights) to Alpaca orders. For each asset, we calculate the target number of shares based on the current price and the target weight.

If we need to buy more shares, we submit a buy order. If we have excess shares, we submit a sell order.
We continue this process until the environment indicates it's done (end of simulation period).

Running the Strategy: Finally, we call the execute_trades function with the loaded MDP model and environment to run the trading strategy.

