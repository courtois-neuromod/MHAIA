import numpy as np
from collections import deque
from typing import List, Dict

from GHAIA.env.scenario import MarioEnv
from GHAIA.wrappers.reward import WrapperHolder, PositionRewardWrapper, ConstantRewardWrapper


class World8(MarioEnv):
    """
    World 8 of Super Mario Bros - The Final World

    The hardest world, featuring all enemy types and maximum difficulty.

    Stages:
    - Level8-1: Gauntlet of all enemy types
    - Level8-2: Challenging underground maze
    - Level8-3: Extreme platforming at height
    - Level8-4: Final castle with the real Bowser

    Success is measured by horizontal progress through the level (x-position).
    The agent receives rewards for moving right and penalties for time passing.
    """

    # Level names for each stage in World 8
    TASKS = ['Level8-1', 'Level8-2', 'Level8-3', 'Level8-4']

    def __init__(self,
                 mario_kwargs: Dict[str, any],
                 reward_position: float = 1.0,
                 penalty_time: float = -0.01,
                 reward_coin: float = 10.0):
        """
        Initialize World 8 environment.

        Args:
            mario_kwargs: Keyword arguments passed to MarioEnv
            reward_position: Scalar for position-based rewards (progress through level)
            penalty_time: Penalty per frame (encourages faster completion)
            reward_coin: Reward for collecting a coin
        """
        super().__init__(**mario_kwargs)
        self.reward_position = reward_position
        self.penalty_time = penalty_time
        self.reward_coin = reward_coin

        # Statistics tracking
        self.max_x_position = 0
        self.total_coins = 0
        self.total_score = 0
        self.frames_elapsed = 0

    def store_statistics(self, game_var_buf: deque) -> None:
        """
        Track statistics during episode.
        """
        self.frames_elapsed += 1

        if len(game_var_buf) == 0:
            return

        current_state = game_var_buf[-1]

        # Track maximum x position reached
        x_pos = current_state.get('xscrollHi', 0) * 256 + current_state.get('xscrollLo', 0)
        self.max_x_position = max(self.max_x_position, x_pos)

        # Track total coins and score
        self.total_coins = current_state.get('coins', 0)
        self.total_score = current_state.get('score', 0)

    def get_success_metric(self) -> float:
        """
        Success is measured by maximum horizontal position reached.
        Further right = better performance.
        """
        return float(self.max_x_position)

    def reward_wrappers_dense(self) -> List[WrapperHolder]:
        """
        Dense reward: Reward for progress + time penalty.
        """
        return [
            WrapperHolder(PositionRewardWrapper, scaler=self.reward_position),
            WrapperHolder(ConstantRewardWrapper, reward=self.penalty_time),
        ]

    def reward_wrappers_sparse(self) -> List[WrapperHolder]:
        """
        Sparse reward: Only reward for progress.
        """
        return [
            WrapperHolder(PositionRewardWrapper, scaler=self.reward_position),
        ]

    @property
    def performance_upper_bound(self) -> float:
        """
        Upper bound for normalization - approximate max position at goal.
        """
        level_bounds = {
            'Level8-1': 3200,
            'Level8-2': 3200,
            'Level8-3': 3200,
            'Level8-4': 3000,
        }
        return float(level_bounds.get(self.env_name, 3200))

    @property
    def performance_lower_bound(self) -> float:
        """
        Lower bound for normalization - minimal progress (random agent).
        """
        return 100.0

    def extra_statistics(self, mode: str = '') -> Dict[str, float]:
        """
        Return additional statistics for logging.
        """
        return {
            f'{mode}/max_x_position': float(self.max_x_position),
            f'{mode}/coins': float(self.total_coins),
            f'{mode}/score': float(self.total_score),
            f'{mode}/frames': float(self.frames_elapsed),
        }

    def clear_episode_statistics(self) -> None:
        """
        Reset statistics at episode start.
        """
        super().clear_episode_statistics()
        self.max_x_position = 0
        self.total_coins = 0
        self.total_score = 0
        self.frames_elapsed = 0
