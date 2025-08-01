# src/environment/reward_schemes.py
# <<< NEW MODULE: Isolates all reward calculation logic >>>

import logging

log = logging.getLogger(__name__)

def calculate_pnl_and_shaping_reward(
    pnl_component: float,
    action: int,
    is_invalid_action: bool,
    consecutive_holds: int,
    trade_occurred: bool,
    hold_penalty_scale: float,
    activity_reward_scale: float,
    max_consecutive_holds: int,
    excessive_hold_penalty: float
) -> tuple[float, bool]:
    """
    Calculates the reward based on PnL and applies shaping penalties/bonuses.
    Returns the final reward and a boolean indicating if the episode terminated due to a penalty.
    """
    reward = pnl_component
    terminated_by_penalty = False

    # Action-based shaping
    if action == 0:  # ACTION_HOLD
        reward -= hold_penalty_scale * consecutive_holds
    else:
        if trade_occurred and pnl_component > 0:
            # Reward profitable activity
            reward += pnl_component * activity_reward_scale
        elif is_invalid_action:
            # Penalize trying to take an invalid action
            reward -= 0.01

    # State-based penalty
    if consecutive_holds > max_consecutive_holds:
        log.warning(f"Exceeded max consecutive holds ({consecutive_holds}/{max_consecutive_holds}). Applying penalty.")
        reward -= excessive_hold_penalty
        terminated_by_penalty = True

    return float(reward), terminated_by_penalty