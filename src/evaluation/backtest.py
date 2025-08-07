# <<< REFACTORED: Runs the backtest simulation and returns raw results. >>>
# <<< DEFINITIVE FIX: Correctly handles the 4-item return from a VecEnv.step() call. >>>

import pandas as pd
import numpy as np
import logging
from tqdm import tqdm
from dataclasses import dataclass, field
from typing import List, Dict, Any

from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.vec_env import DummyVecEnv

log = logging.getLogger(__name__)

@dataclass
class BacktestResult:
    """A structured object to hold the raw results of a backtest."""
    trades: List[Dict[str, Any]] = field(default_factory=list)
    equity_curve: pd.DataFrame = field(default_factory=pd.DataFrame)
    positions: pd.DataFrame = field(default_factory=pd.DataFrame)
    initial_balance: float = 0.0
    final_balance: float = 0.0
    metrics: Dict[str, Any] = field(default_factory=dict)

class BacktestRunner:
    """
    Executes the backtest loop for a given agent and environment.
    Its sole responsibility is to produce a BacktestResult.
    """
    def __init__(self, model: BaseAlgorithm, env: DummyVecEnv, deterministic: bool):
        self.model = model
        self.env = env
        self.deterministic = deterministic
        self.initial_balance = self.env.get_attr('account')[0].initial_balance

    def run(self) -> BacktestResult:
        """Executes the main backtest loop."""
        obs = self.env.reset()
        terminated, truncated = False, False
        
        timestamps, portfolio_values, positions_held = [], [], []
        trades, current_trade = [], {}
        
        initial_step = self.env.get_attr('current_step')[0]
        initial_timestamp = self.env.get_attr('df')[0].index[initial_step]
        initial_portfolio_value = self.env.get_attr('account')[0].portfolio_value(self.env.get_attr('df')[0]['close'].iloc[initial_step])
        initial_position = self.env.get_attr('account')[0].position

        timestamps.append(initial_timestamp)
        portfolio_values.append(initial_portfolio_value)
        positions_held.append(initial_position)

        pbar = tqdm(total=len(self.env.get_attr('df')[0]) - self.env.get_attr('current_step')[0], desc="Backtest Progress")
        
        info = {} # Initialize info dict
        while not (terminated or truncated):
            prev_position = self.env.get_attr('account')[0].position
            action, _ = self.model.predict(obs, deterministic=self.deterministic)
            
            # <<< THE FIX IS HERE: Use the 4-item return from VecEnv.step() >>>
            obs, rewards, dones, infos = self.env.step(action)
            
            info = infos[0]
            # For a single env, dones[0] is True if the episode is over (terminated or truncated)
            terminated = dones[0] 
            # SB3 puts the truncation signal inside the info dict
            truncated = info.get("TimeLimit.truncated", False) and terminated

            timestamps.append(info.get('timestamp', pd.NaT))
            portfolio_values.append(info.get('portfolio_value', portfolio_values[-1]))
            positions_held.append(info.get('position', prev_position))
            
            current_trade = self._log_trade_transitions(info, prev_position, action[0], current_trade, trades)

            pbar.update(1)
        pbar.close()

        equity_df = pd.DataFrame({'portfolio_value': portfolio_values}, index=pd.to_datetime(timestamps, utc=True))
        positions_df = pd.DataFrame({'position': positions_held}, index=pd.to_datetime(timestamps, utc=True))

        return BacktestResult(
            trades=trades,
            equity_curve=equity_df,
            positions=positions_df,
            initial_balance=self.initial_balance,
            final_balance=portfolio_values[-1],
            metrics=info # Use the info dict from the last step of the loop
        )

    def _log_trade_transitions(self, info: Dict, prev_pos: int, action: int, current_trade: Dict, trades: List) -> Dict:
        """Manages the lifecycle of a trade dictionary."""
        current_pos = info.get('position', prev_pos)
        
        if current_trade and prev_pos != 0 and current_pos != prev_pos:
            current_trade.update({
                'exit_ts': info.get('timestamp'),
                'exit_price': info.get('market_price'),
                'pnl': info.get('realized_pnl_on_close', 0.0),
                'exit_action': action
            })
            trades.append(current_trade.copy())
            log.info(f"Trade CLOSED. Type: {current_trade.get('direction')}, PnL: {current_trade.get('pnl', 0.0):.2f}")
            current_trade = {}

        if prev_pos != current_pos and current_pos != 0:
            current_trade = {
                'entry_ts': info.get('timestamp'),
                'entry_price': info.get('entry_price'),
                'direction': 'long' if current_pos > 0 else 'short',
                'contracts': abs(info.get('contracts_held', 0)),
                'entry_action': action
            }
            log.info(f"Trade OPENED. Type: {current_trade['direction']}, Entry: {current_trade.get('entry_price', 'N/A'):.2f}")

        return current_trade
