# src/environment/account_state.py
# <<< NEW MODULE: Manages the agent's financial portfolio state >>>

import numpy as np
import logging

log = logging.getLogger(__name__)

class AccountState:
    """
    Encapsulates all financial state and position management for the trading agent.
    This class acts as the agent's brokerage account.
    """
    def __init__(self, initial_balance: float, point_value: float, commission_per_contract: float):
        self.initial_balance = initial_balance
        self.point_value = point_value
        self.commission_per_contract = commission_per_contract
        self.reset()

    def reset(self):
        """Resets the account to its initial state."""
        self.balance = self.initial_balance
        self.contracts_held = 0
        self.position = 0  # 0: flat, 1: long, -1: short
        self.entry_price = 0.0
        self.steps_in_trade = 0
        log.debug("AccountState reset to initial values.")

    def calculate_unrealized_pnl(self, current_price: float) -> float:
        """Calculates the PnL of the current open position."""
        if self.position == 0 or np.isnan(self.entry_price) or self.entry_price == 0.0:
            return 0.0
        if self.position == 1:
            return (current_price - self.entry_price) * self.contracts_held * self.point_value
        elif self.position == -1:
            return (self.entry_price - current_price) * abs(self.contracts_held) * self.point_value
        return 0.0

    def portfolio_value(self, current_price: float) -> float:
        """Calculates the total equity of the account (balance + unrealized PnL)."""
        return self.balance + self.calculate_unrealized_pnl(current_price)

    def open_position(self, side: int, execution_price: float, trade_size: int) -> float:
        """Opens a new position and deducts entry commission."""
        self.position = side
        self.contracts_held = trade_size * self.position
        self.entry_price = execution_price
        self.steps_in_trade = 0

        commission_cost = self.commission_per_contract * trade_size
        self.balance -= commission_cost
        return commission_cost

    def close_position(self, execution_price: float) -> tuple[float, float]:
        """Closes the current position, calculates PnL, and updates the balance."""
        if self.position == 0:
            return 0.0, 0.0

        num_contracts_closed = abs(self.contracts_held)
        commission_cost = self.commission_per_contract * num_contracts_closed

        realized_pnl = 0.0
        if self.position == 1:
            realized_pnl = (execution_price - self.entry_price) * num_contracts_closed * self.point_value
        elif self.position == -1:
            realized_pnl = (self.entry_price - execution_price) * num_contracts_closed * self.point_value

        net_pnl_trade = realized_pnl - commission_cost
        self.balance += net_pnl_trade

        # Reset position state
        self.contracts_held = 0
        self.position = 0
        self.entry_price = 0.0
        self.steps_in_trade = 0

        return net_pnl_trade, commission_cost