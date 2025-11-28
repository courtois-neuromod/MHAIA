from typing import Callable

import numpy as np
from gymnasium import RewardWrapper


class WrapperHolder:
    """
    A wrapper holder stores a reward wrapper with its respective keyword arguments.
    """

    def __init__(self, wrapper_class, **kwargs):
        self.wrapper_class = wrapper_class
        self.kwargs = kwargs


class ConstantRewardWrapper(RewardWrapper):
    """
    Reward the agent with a constant reward.
    """

    def __init__(self, env, reward: float):
        super(ConstantRewardWrapper, self).__init__(env)
        self.rew = reward

    def reward(self, reward):
        return reward + self.rew


class BooleanVariableRewardWrapper(RewardWrapper):
    """
    Reward the agent if a game variable is true.
    """

    def __init__(self, env, reward: float, var_name: str):
        super(BooleanVariableRewardWrapper, self).__init__(env)
        self.rew = reward
        self.var_name = var_name

    def reward(self, reward):
        game_variable = self.env.get_state_variable(self.var_name)
        if game_variable:
            reward += self.rew
        return reward


class StateVariableRewardWrapper(RewardWrapper):
    """
    Reward the agent for a change in a game state variable. The agent is considered to have changed a game variable
    if its value differs from the previous frame value.
    """

    def __init__(self, env, reward: float, var_name: str, decrease: bool = False):
        super(StateVariableRewardWrapper, self).__init__(env)
        self.rew = reward
        self.var_name = var_name
        self.decrease = decrease

    def reward(self, reward):
        if len(self.game_variable_buffer) < 2:
            return reward

        vars_cur = self.game_variable_buffer[-1]
        vars_prev = self.game_variable_buffer[-2]

        var_cur = vars_cur.get(self.var_name, 0)
        var_prev = vars_prev.get(self.var_name, 0)

        if not self.decrease and var_cur > var_prev or self.decrease and var_cur < var_prev:
            reward += self.rew
        return reward


class CumulativeVariableRewardWrapper(RewardWrapper):
    """
    Cumulatively reward the agent for a change in a game variable. The agent is considered to have changed a game
    variable if its value is higher than it was in the previous frame.
    """

    def __init__(self, env, reward: float, var_name: str, decrease: bool = False, maintain: bool = False):
        super(CumulativeVariableRewardWrapper, self).__init__(env)
        self.rew = reward
        self.var_name = var_name
        self.decrease = decrease
        self.maintain = maintain
        self.cum_rew = 0

    def reward(self, reward):
        if len(self.game_variable_buffer) < 2:
            return reward

        vars_cur = self.game_variable_buffer[-1]
        vars_prev = self.game_variable_buffer[-2]

        var_cur = vars_cur.get(self.var_name, 0)
        var_prev = vars_prev.get(self.var_name, 0)

        if self.maintain and var_cur == var_prev or not self.decrease and var_cur > var_prev or self.decrease and var_cur < var_prev:
            self.cum_rew += self.rew
            reward += self.cum_rew
        else:
            self.cum_rew = 0
        return reward


class ProportionalVariableRewardWrapper(RewardWrapper):
    """
    Proportionally reward the agent for a change in a game variable. The agent is considered to have changed a game
    variable if its value is higher than it was in the previous frame.
    """

    def __init__(self, env, scaler: float, var_name: str, keep_lb: bool = False):
        super(ProportionalVariableRewardWrapper, self).__init__(env)
        self.scaler = scaler
        self.var_name = var_name
        self.keep_lb = keep_lb
        self.lower_bound = -np.inf

    def reward(self, reward):
        if len(self.game_variable_buffer) < 2:
            self.lower_bound = -np.inf
            return reward

        var_cur = self.game_variable_buffer[-1].get(self.var_name, 0)
        var_prev = self.game_variable_buffer[-2].get(self.var_name, 0)

        if not self.keep_lb or self.keep_lb and var_cur > self.lower_bound:
            reward = self.scaler * (var_cur - var_prev)
        self.lower_bound = max(var_cur, self.lower_bound) if self.keep_lb else 0
        return reward


class UserVariableRewardWrapper(RewardWrapper):
    """
    Reward the agent for a change in a user variable. The agent is considered to have changed a user variable if its
    value is higher than it was in the previous frame.
    """

    def __init__(self, env, reward: float, var_name: str, decrease: bool = False,
                 update_callback: Callable = None):
        super(UserVariableRewardWrapper, self).__init__(env)
        self.rew = reward
        self.var_name = var_name
        self.decrease = decrease
        self.update_callback = update_callback

    def reward(self, reward):
        var_cur = self.get_state_variable(self.var_name)
        var_prev = self.get_and_update_user_var(self.var_name)

        if not self.decrease and var_cur > var_prev or self.decrease and var_cur < var_prev:
            reward += self.rew
        return reward


class PositionRewardWrapper(RewardWrapper):
    """
    Reward the agent for horizontal movement (progress through the level).
    Uses xscrollLo and xscrollHi for precise position tracking.
    """

    def __init__(self, env, scaler: float = 0.1):
        super(PositionRewardWrapper, self).__init__(env)
        self.scaler = scaler
        self.prev_position = 0

    def reward(self, reward):
        if len(self.game_variable_buffer) < 2:
            # Initialize position on first step
            vars_cur = self.game_variable_buffer[-1] if len(self.game_variable_buffer) > 0 else {}
            xscroll_lo = vars_cur.get('xscrollLo', 0)
            xscroll_hi = vars_cur.get('xscrollHi', 0)
            self.prev_position = xscroll_hi * 256 + xscroll_lo
            return reward

        vars_cur = self.game_variable_buffer[-1]
        vars_prev = self.game_variable_buffer[-2]

        # Calculate current and previous X positions (combining Hi and Lo bytes)
        x_cur = vars_cur.get('xscrollHi', 0) * 256 + vars_cur.get('xscrollLo', 0)
        x_prev = vars_prev.get('xscrollHi', 0) * 256 + vars_prev.get('xscrollLo', 0)

        # Reward for moving right (positive progress)
        position_delta = max(0, x_cur - x_prev)
        reward += position_delta * self.scaler

        self.prev_position = x_cur
        return reward


class ScoreRewardWrapper(RewardWrapper):
    """
    Reward the agent for increasing the game score.
    """

    def __init__(self, env, scaler: float = 0.001):
        super(ScoreRewardWrapper, self).__init__(env)
        self.scaler = scaler

    def reward(self, reward):
        if len(self.game_variable_buffer) < 2:
            return reward

        vars_cur = self.game_variable_buffer[-1]
        vars_prev = self.game_variable_buffer[-2]

        score_cur = vars_cur.get('score', 0)
        score_prev = vars_prev.get('score', 0)

        score_delta = max(0, score_cur - score_prev)
        reward += score_delta * self.scaler
        return reward


class CoinRewardWrapper(RewardWrapper):
    """
    Reward the agent for collecting coins.
    """

    def __init__(self, env, reward: float = 1.0):
        super(CoinRewardWrapper, self).__init__(env)
        self.rew = reward

    def reward(self, reward):
        if len(self.game_variable_buffer) < 2:
            return reward

        vars_cur = self.game_variable_buffer[-1]
        vars_prev = self.game_variable_buffer[-2]

        coins_cur = vars_cur.get('coins', 0)
        coins_prev = vars_prev.get('coins', 0)

        if coins_cur > coins_prev:
            reward += self.rew
        return reward


class TimeRewardWrapper(RewardWrapper):
    """
    Penalty for time passing (encourages faster completion).
    """

    def __init__(self, env, penalty: float = -0.01):
        super(TimeRewardWrapper, self).__init__(env)
        self.penalty = penalty

    def reward(self, reward):
        # Apply constant time penalty
        return reward + self.penalty


class DeathPenaltyWrapper(RewardWrapper):
    """
    Penalty for losing a life.
    """

    def __init__(self, env, penalty: float = -10.0):
        super(DeathPenaltyWrapper, self).__init__(env)
        self.penalty = penalty
        self.prev_lives = None

    def reward(self, reward):
        if len(self.game_variable_buffer) == 0:
            return reward

        vars_cur = self.game_variable_buffer[-1]
        lives_cur = vars_cur.get('lives', 0)

        if self.prev_lives is not None and lives_cur < self.prev_lives:
            reward += self.penalty

        self.prev_lives = lives_cur
        return reward


# Legacy aliases for backward compatibility
GameVariableRewardWrapper = StateVariableRewardWrapper
MovementRewardWrapper = PositionRewardWrapper


class LocationVariableRewardWrapper(RewardWrapper):
    """
    Reward the agent for traversing a certain distance. The agent is considered to have traversed a distance if its
    location is further away from the starting location than it was in the previous frame.
    """

    def __init__(self, env, x_var_name: str, y_var_name: str, x_start: float, y_start: float, scaler: float = 0.1):
        super(LocationVariableRewardWrapper, self).__init__(env)
        self.x_var_name = x_var_name
        self.y_var_name = y_var_name
        self.x_start = x_start
        self.y_start = y_start
        self.scaler = scaler

    def reward(self, reward):
        if len(self.game_variable_buffer) < 2:
            return reward

        vars_cur = self.game_variable_buffer[-1]
        vars_prev = self.game_variable_buffer[-2]

        x_cur = vars_cur.get(self.x_var_name, 0)
        y_cur = vars_cur.get(self.y_var_name, 0)
        x_prev = vars_prev.get(self.x_var_name, 0)
        y_prev = vars_prev.get(self.y_var_name, 0)

        x_diff = max(0, abs(x_cur - self.x_start) - abs(x_prev - self.x_start))
        y_diff = max(0, abs(y_cur - self.y_start) - abs(y_prev - self.y_start))
        return reward + self.scaler * (x_diff + y_diff)


class PlatformReachedRewardWrapper(RewardWrapper):
    """
    Reward the agent for reaching a platform. The agent is considered to be on a platform if its height is higher than
    the highest height it was on in the last n frames.
    """

    def __init__(self, env, reward: float, z_var_name: str = 'player_y_pos'):
        super(PlatformReachedRewardWrapper, self).__init__(env)
        self.z_var_name = z_var_name
        self.rew = reward

    def reward(self, reward):
        if len(self.game_variable_buffer) < 2:
            return reward

        vars_cur = self.game_variable_buffer[-1]
        height_cur = vars_cur.get(self.z_var_name, 0)
        heights_prev = [game_vars.get(self.z_var_name, 0) for game_vars in self.game_variable_buffer]

        # Check whether the agent was lower in the last n frames and is now higher
        if height_cur > max(heights_prev[:-1]):
            reward += self.rew
        return reward


class GoalRewardWrapper(RewardWrapper):
    """
    Reward the agent for reaching a goal. The agent is considered to have reached the goal if the value of a game
    variable is higher than a given threshold.
    """

    def __init__(self, env, reward: float, goal: float, var_name: str):
        super(GoalRewardWrapper, self).__init__(env)
        self.rew = reward
        self.goal = goal
        self.var_name = var_name

    def reward(self, reward):
        if len(self.game_variable_buffer) == 0:
            return reward

        vars_cur = self.game_variable_buffer[-1]
        var_cur = vars_cur.get(self.var_name, 0)

        if var_cur > self.goal:
            reward += self.rew
        return reward
