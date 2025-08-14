# <<< SAC-READY v2: Integrates the more aggressive shaped reward scheme. >>>

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
import logging
from omegaconf import DictConfig
from typing import Dict, Tuple, Optional, List

from .account_state import AccountState
from .reward_schemes import calculate_pnl_and_shaping_reward, calculate_sac_shaped_reward

log = logging.getLogger(__name__)

class FuturesTradingEnv(gym.Env):
    metadata = {'render_modes': ['human', None]}
    ACTION_HOLD, ACTION_ENTER_LONG, ACTION_ENTER_SHORT, ACTION_EXIT_POSITION = 0, 1, 2, 3

    def __init__(self, df: pd.DataFrame, feature_cols: List[str], cfg: DictConfig):
        super().__init__()
        self.cfg_main = cfg
        self.env_cfg = cfg.environment
        log.info("Initializing FuturesTradingEnv (SAC Aggressive Reward Version)...")

        self._validate_inputs(df, feature_cols)
        self.df = self._prepare_dataframe(df)
        
        if 'source_regime' in self.df.columns:
            self.is_regime_boundary = (self.df['source_regime'] != self.df['source_regime'].shift(1)).fillna(False)
        else:
            self.is_regime_boundary = pd.Series(False, index=self.df.index)
        
        self.feature_cols = feature_cols
        self.n_features = len(feature_cols)
        self.lookback_window = int(self.env_cfg.lookback_window)
        self._start_step = self.lookback_window - 1

        self.account = AccountState(
            initial_balance=float(self.env_cfg.initial_balance),
            point_value=float(self.env_cfg.point_value),
            commission_per_contract=float(self.env_cfg.commission_per_contract)
        )

        self._configure_simulation_params()
        self._define_spaces()
        self.reset()

    def _validate_inputs(self, df, feature_cols):
        if df.empty: raise ValueError("Input DataFrame 'df' cannot be empty.")
        if not feature_cols: raise ValueError("Feature list 'feature_cols' cannot be empty.")
        if not all(c in df.columns for c in ['open', 'high', 'low', 'close']): raise ValueError("DataFrame needs OHLC columns.")
        if not all(c in df.columns for c in feature_cols): raise ValueError("DataFrame missing specified feature columns.")
        if not pd.api.types.is_datetime64_any_dtype(df.index): raise ValueError("DataFrame index must be a DatetimeIndex.")

    def _prepare_dataframe(self, df):
        df_copy = df.copy()
        if df_copy.index.tzinfo is None or str(df_copy.index.tzinfo).upper() != 'UTC':
            df_copy = df_copy.tz_localize('UTC') if df_copy.index.tzinfo is None else df_copy.tz_convert('UTC')
        df_copy['day'] = df_copy.index.date
        return df_copy

    def _configure_simulation_params(self):
        self.tick_size = float(self.env_cfg.tick_size)
        self.stop_loss_ticks = int(self.env_cfg.get('stop_loss_ticks', 0))
        if self.stop_loss_ticks > 0:
            self.stop_loss_points = self.stop_loss_ticks * self.tick_size
            log.info(f"Hard stop-loss enabled at {self.stop_loss_ticks} ticks ({self.stop_loss_points} points).")
        else:
            self.stop_loss_points = 0
            log.info("Hard stop-loss is disabled.")
        self.add_bid_ask_spread = bool(self.env_cfg.get('add_bid_ask_spread', False))
        self.spread_ticks_min = int(self.env_cfg.get('spread_ticks_min', 1))
        self.spread_ticks_max = int(self.env_cfg.get('spread_ticks_max', 2))
        self.max_slippage_ticks = float(self.env_cfg.max_slippage_points) / self.tick_size
        self.base_contracts = int(self.env_cfg.get('position_sizing.base_contracts', 1))
        self.activity_reward_scale = float(self.env_cfg.get('activity_reward_scale', 0.0))
        self.hold_penalty_scale = float(self.env_cfg.get('hold_penalty_scale', 0.0))
        self.win_bonus_scale = float(self.env_cfg.get('win_bonus_scale', 0.0))
        self.loss_penalty_scale = float(self.env_cfg.get('loss_penalty_scale', 0.0))
        self.unrealized_loss_penalty_scale = float(self.env_cfg.get('unrealized_loss_penalty_scale', 0.0))
        self.max_consecutive_holds_limit = int(self.env_cfg.get("max_consecutive_holds_limit", 900))
        self.excessive_hold_penalty = float(self.env_cfg.get("excessive_hold_penalty", 0.5))
        self.reward_scheme = self.env_cfg.get('reward_scheme', 'ppo_legacy')

    def _define_spaces(self):
        self.action_space = spaces.Discrete(4)
        self.account_info_features = 4
        feature_box_shape = (self.lookback_window, self.n_features + self.account_info_features)
        low = np.full(feature_box_shape, -np.inf, dtype=np.float32)
        high = np.full(feature_box_shape, np.inf, dtype=np.float32)
        low[:, -self.account_info_features:] = [0.0, -1.1, -5.0, 0.0]
        high[:, -self.account_info_features:] = [20.0, 1.1, 5.0, 1.0]
        self.observation_space = spaces.Dict({"features": spaces.Box(low=low, high=high, dtype=np.float32)})

    def _get_current_price(self) -> float:
        return self.df['close'].iloc[self.current_step]

    def _get_obs(self) -> Dict[str, np.ndarray]:
        end_idx = self.current_step + 1
        start_idx = end_idx - self.lookback_window
        features_part = self.df[self.feature_cols].iloc[start_idx:end_idx].values
        balance_norm = self.account.balance / self.account.initial_balance
        unrealized_pnl = self.account.calculate_unrealized_pnl(self._get_current_price())
        pnl_norm_factor = self.account.initial_balance * 0.02
        unrealized_pnl_norm = np.clip(unrealized_pnl / pnl_norm_factor, -5.0, 5.0)
        trade_duration_norm = np.clip(self.account.steps_in_trade / self.env_cfg.max_episode_steps, 0.0, 1.0)
        account_info_slice = np.array([balance_norm, float(self.account.position), unrealized_pnl_norm, trade_duration_norm], dtype=np.float32)
        account_info_repeated = np.tile(account_info_slice, (self.lookback_window, 1))
        obs_array = np.hstack((features_part, account_info_repeated)).astype(np.float32)
        return {"features": np.clip(obs_array, self.observation_space['features'].low, self.observation_space['features'].high)}

    def _get_info(self) -> dict:
        info = {"step": self.current_step, "timestamp": self.df.index[self.current_step], "balance": self.account.balance, "contracts_held": self.account.contracts_held, "position": self.account.position, "entry_price": self.account.entry_price, "market_price": self._get_current_price(), "portfolio_value": self.account.portfolio_value(self._get_current_price()), "trade_occurred_this_step": self._trade_occurred_this_step, "realized_pnl_on_close": self._info_realized_pnl_on_close, "consecutive_holds": self._consecutive_holds, "steps_in_trade": self.account.steps_in_trade}
        self._info_realized_pnl_on_close = None
        return info

    def _get_agent_execution_price(self, order_side: int) -> Optional[float]:
        next_step_idx = self.current_step + 1
        if next_step_idx >= len(self.df): return None
        base_price = self.df['open'].iloc[next_step_idx]
        if self.add_bid_ask_spread:
            spread_in_ticks = self.np_random.uniform(self.spread_ticks_min, self.spread_ticks_max)
            spread_in_points = (spread_in_ticks * self.tick_size) / 2.0
            base_price += spread_in_points if order_side == 1 else -spread_in_points
        slippage_in_ticks = self.np_random.uniform(0, self.max_slippage_ticks)
        slippage_in_points = slippage_in_ticks * self.tick_size
        return base_price + slippage_in_points if order_side == 1 else base_price - slippage_in_points

    def _is_trading_forbidden(self) -> bool:
        if self.is_regime_boundary.iloc[self.current_step]: return True
        ts = self.df.index[self.current_step]
        if ts.weekday() == 4 and ts.hour >= 20: return True
        if self.current_step >= len(self.df) - 1: return True
        return False

    def _execute_agent_trade(self, action: int):
        self._is_invalid_action = False
        self._trade_occurred_this_step = False
        trade_size = self.base_contracts
        if action == self.ACTION_HOLD: return
        if action in [self.ACTION_ENTER_LONG, self.ACTION_ENTER_SHORT]:
            if self._is_trading_forbidden():
                self._is_invalid_action = True
                return
            if self.account.position == 0:
                order_side = 1 if action == self.ACTION_ENTER_LONG else -1
                if (price := self._get_agent_execution_price(order_side)) is not None:
                    self.account.open_position(order_side, price, trade_size)
                    self._trade_occurred_this_step = True
            else: self._is_invalid_action = True
            return
        if action == self.ACTION_EXIT_POSITION:
            if self.account.position != 0:
                order_side = -self.account.position
                if (price := self._get_agent_execution_price(order_side)) is not None:
                    pnl, _ = self.account.close_position(price)
                    self._info_realized_pnl_on_close = pnl
                    self._trade_occurred_this_step = True
            else: self._is_invalid_action = True
            return

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[Dict[str, np.ndarray], dict]:
        super().reset(seed=seed)
        self.account.reset()
        self.current_step = self._start_step
        self._consecutive_holds = 0
        self._trade_occurred_this_step = False
        self._is_invalid_action = False
        self._info_realized_pnl_on_close = None
        return self._get_obs(), self._get_info()

    def step(self, action: int) -> Tuple[Dict[str, np.ndarray], float, bool, bool, dict]:
        portfolio_value_before = self.account.portfolio_value(self._get_current_price())
        position_was_flat = self.account.position == 0
        
        stop_loss_hit = self._check_for_stop_loss()
        forced_exit_occurred = self._handle_forced_exits()
        if not forced_exit_occurred and not stop_loss_hit:
            self._execute_agent_trade(action)
        else:
            self._trade_occurred_this_step = True
            self._is_invalid_action = False
        
        if self.account.position != 0: self.account.steps_in_trade += 1
        if action == self.ACTION_HOLD and not (forced_exit_occurred or stop_loss_hit):
            self._consecutive_holds += 1
        else:
            self._consecutive_holds = 0

        portfolio_value_after = self.account.portfolio_value(self._get_current_price())
        pnl_component = portfolio_value_after - portfolio_value_before
        
        if self.reward_scheme == 'sac_shaped':
            realized_pnl = self._info_realized_pnl_on_close or 0.0
            trade_closed = self._info_realized_pnl_on_close is not None
            volatility_feature = self.df['high_low_pct'].iloc[self.current_step]
            current_unrealized_pnl = self.account.calculate_unrealized_pnl(self._get_current_price())
            reward = calculate_sac_shaped_reward(
                unrealized_pnl_change=pnl_component, realized_pnl_on_close=realized_pnl,
                current_unrealized_pnl=current_unrealized_pnl,
                action_taken=action, position_was_flat=position_was_flat,
                trade_closed_this_step=trade_closed, volatility=volatility_feature,
                activity_bonus_scale=self.activity_bonus_scale, hold_penalty_scale=self.hold_penalty_scale,
                win_bonus_scale=self.win_bonus_scale, loss_penalty_scale=self.loss_penalty_scale,
                unrealized_loss_penalty_scale=self.unrealized_loss_penalty_scale
            )
            terminated_by_penalty = False
        else:
            reward, terminated_by_penalty = calculate_pnl_and_shaping_reward(
                pnl_component, action, self._is_invalid_action, self._consecutive_holds,
                self._trade_occurred_this_step, self.hold_penalty_scale, self.activity_reward_scale,
                self.max_consecutive_holds_limit, self.excessive_hold_penalty
            )

        self.current_step += 1
        terminated = terminated_by_penalty or (self.current_step >= len(self.df) - 1)
        truncated = self.current_step >= len(self.df) - 1
        return self._get_obs(), reward, terminated, truncated, self._get_info()

    def _check_for_stop_loss(self) -> bool:
        if self.account.position == 0 or self.stop_loss_points == 0: return False
        current_bar = self.df.iloc[self.current_step]
        if self.account.position == 1:
            loss_price = self.account.entry_price - self.stop_loss_points
            if current_bar['low'] <= loss_price:
                self._force_close_position_now(loss_price)
                return True
        elif self.account.position == -1:
            loss_price = self.account.entry_price + self.stop_loss_points
            if current_bar['high'] >= loss_price:
                self._force_close_position_now(loss_price)
                return True
        return False

    def _force_close_position_now(self, execution_price: float):
        if self.account.position == 0: return
        pnl, _ = self.account.close_position(execution_price)
        self._info_realized_pnl_on_close = pnl
        log.info(f"Forced close executed at price: {execution_price:.2f}, PnL: {pnl:.2f}")

    def _handle_forced_exits(self) -> bool:
        action_forced = False
        if self.account.position != 0 and self.is_regime_boundary.iloc[self.current_step]:
            exit_price = self.df['close'].iloc[self.current_step - 1]
            self._force_close_position_now(exit_price)
            action_forced = True
            return action_forced
        ts = self.df.index[self.current_step]
        if self.account.position != 0 and ts.weekday() == 4 and ts.hour >= 20:
            self._force_close_position_now(self._get_current_price())
            action_forced = True
        if self.account.position != 0 and (self.current_step >= len(self.df) - 1):
            self._force_close_position_now(self._get_current_price())
            action_forced = True
        return action_forced

    def close(self):
        log.info("Closing FuturesTradingEnv.")
