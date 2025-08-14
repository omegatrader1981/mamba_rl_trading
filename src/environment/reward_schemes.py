# <<< SAC-READY v2: Implements a more aggressive shaped reward with a per-step loss penalty. >>>

import logging
import numpy as np

log = logging.getLogger(__name__)

def calculate_sac_shaped_reward(
    # --- Core PnL & State ---
    unrealized_pnl_change: float,
    realized_pnl_on_close: float,
    current_unrealized_pnl: float, # <<< NEW INPUT
    
    # --- Agent's Action & State ---
    action_taken: int,
    position_was_flat: bool,
    trade_closed_this_step: bool,
    
    # --- Market Context ---
    volatility: float,
    
    # --- Configurable Hyperparameters ---
    activity_bonus_scale: float,
    hold_penalty_scale: float,
    win_bonus_scale: float,
    loss_penalty_scale: float,
    unrealized_loss_penalty_scale: float # <<< NEW HYPERPARAMETER
) -> float:
    """
    Calculates a shaped reward for the SAC agent. Version 2 adds a continuous
    penalty for holding an unrealized loss.
    """
    ACTION_HOLD = 0
    reward = unrealized_pnl_change

    # --- Activity & Inaction Shaping ---
    if action_taken == ACTION_HOLD and position_was_flat:
        reward -= hold_penalty_scale * volatility
    elif action_taken != ACTION_HOLD and position_was_flat:
        reward += activity_bonus_scale * volatility

    # <<< THE KEY NEW LOGIC IS HERE >>>
    # --- Per-Step Penalty for Holding an Unrealized Loss ---
    if current_unrealized_pnl < 0:
        # This provides a constant, negative reinforcement at every step a trade is losing.
        reward -= unrealized_loss_penalty_scale * abs(current_unrealized_pnl)
        log.debug(f"UNREALIZED LOSS PENALTY APPLIED: {-unrealized_loss_penalty_scale * abs(current_unrealized_pnl):.4f}")

    # --- Asymmetric PnL Shaping on Trade Close ---
    if trade_closed_this_step:
        if realized_pnl_on_close > 0:
            reward += win_bonus_scale * realized_pnl_on_close
            log.debug(f"WIN BONUS APPLIED: {win_bonus_scale * realized_pnl_on_close:.4f}")
        elif realized_pnl_on_close < 0:
            reward -= loss_penalty_scale * abs(realized_pnl_on_close)
            log.debug(f"LOSS PENALTY APPLIED: {-loss_penalty_scale * abs(realized_pnl_on_close):.4f}")

    return float(reward)

# The old PPO reward function remains for backward compatibility
def calculate_pnl_and_shaping_reward(
    pnl_component: float, action: int, is_invalid_action: bool,
    consecutive_holds: int, trade_occurred: bool, hold_penalty_scale: float,
    activity_reward_scale: float, max_consecutive_holds: int, excessive_hold_penalty: float
) -> tuple[float, bool]:
    reward = pnl_component
    terminated_by_penalty = False
    if action == 0:
        reward -= hold_penalty_scale * consecutive_holds
    else:
        if trade_occurred and pnl_component > 0:
            reward += pnl_component * activity_reward_scale
        elif is_invalid_action:
            reward -= 0.01
    if consecutive_holds > max_consecutive_holds:
        reward -= excessive_hold_penalty
        terminated_by_penalty = True
    return float(reward), terminated_by_penalty
