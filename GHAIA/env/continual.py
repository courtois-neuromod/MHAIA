from typing import Any, Dict, List, Tuple, Optional, Union

import gymnasium
import numpy as np
from numpy import ndarray

from GHAIA.env.base import BaseEnv
from GHAIA.env.builder import make_env
from GHAIA.env.scenario import MarioEnv
from GHAIA.utils.config import Sequence, sequence_scenarios, sequence_tasks, scenario_config


class ContinualLearningEnv(BaseEnv):
    """
    A class for creating a continual learning environment composed of a sequence of Mario environments,
    suitable for task-incremental learning.

    IMPORTANT: Uses lazy initialization to work around stable-retro's limitation of one emulator per process.
    Environments are created on-demand and properly closed when switching tasks.

    Attributes:
        steps_per_env (int): The number of steps to be executed in each environment before switching to the next.
        envs (List[MarioEnv]): A list of Mario environment instances (lazily initialized).
        _num_tasks (int): The total number of tasks (environments) in the continual learning setup.
        steps (int): The total number of steps across all environments.
        cur_seq_idx (int): The current index of the active environment in the sequence.
        cur_step (int): The current step counter across the entire sequence of environments.

    Args:
        sequence (Sequence): An enumeration specifying the sequence of environments to be included.
        steps_per_env (int, optional): The number of steps for each environment in the sequence. Defaults to 200,000.
        start_from (int, optional): The starting index within the sequence of environments. Defaults to 0.
        random_order (bool, optional): If set to True, the order of environments in the sequence is randomized. Defaults to False.
        scenario_config (List[Dict[str, any]], optional): A list of dictionaries with specific keyword arguments for each scenario in the sequence.
        mario_config (Dict[str, any], optional): Common keyword arguments applicable to all Mario environments in the sequence.
        wrapper_config (Dict[str, any], optional): Configuration for observation and reward wrappers to be applied to each environment.
    """

    def __init__(self,
                 sequence: Sequence,
                 steps_per_env: int = 2e5,
                 start_from: int = 0,
                 random_order: bool = False,
                 scenario_config: List[Dict[str, any]] = None,
                 mario_config: Dict[str, any] = None,
                 wrapper_config: Dict[str, any] = None,
                 ):
        self.steps_per_env = steps_per_env

        # Store configuration for lazy initialization
        self.sequence = sequence
        self.scenarios = sequence_scenarios[sequence]
        self.task_names = sequence_tasks[sequence]
        self.scenario_config = scenario_config
        self.mario_config = mario_config
        self.wrapper_config = wrapper_config

        # Lazy initialization: envs will be created on-demand
        self._num_tasks = len(self.scenarios)
        self.envs = [None] * self._num_tasks  # Placeholder list
        self._current_env = None  # Currently active environment

        self.steps = steps_per_env * self.num_tasks
        self.cur_seq_idx = start_from
        self.cur_step = 0

        # Initialize the first environment
        self._initialize_env(start_from)

    def _initialize_env(self, idx: int) -> None:
        """
        Initialize or switch to environment at given index.
        Closes previous environment if it exists (critical for stable-retro).
        """
        # Close current environment if it exists
        if self._current_env is not None:
            try:
                self._current_env.close()
            except Exception as e:
                print(f"Warning: Error closing environment: {e}")

        # Create new environment
        scenario = self.scenarios[idx]
        task = self.task_names[idx]
        scenario_kwargs = self.scenario_config[idx] if self.scenario_config else {}

        self.envs[idx] = make_env(
            scenario,
            task,
            idx,
            scenario_kwargs,
            self.mario_config,
            self.wrapper_config
        )
        self._current_env = self.envs[idx]

    def _check_steps_bound(self) -> None:
        if self.cur_step >= self.steps:
            raise RuntimeError("Steps limit exceeded for ContinualLearningEnv!")

    def get_active_env(self) -> MarioEnv:
        return self._current_env

    @property
    def name(self) -> str:
        return "ContinualLearningEnv"

    @property
    def task(self) -> str:
        return self.get_active_env().name

    @property
    def task_id(self) -> int:
        return self.cur_seq_idx

    @property
    def num_tasks(self) -> int:
        return self._num_tasks

    @property
    def action_space(self) -> gymnasium.spaces.Discrete:
        return self._current_env.action_space

    @property
    def observation_space(self) -> gymnasium.Space:
        return self._current_env.observation_space

    @property
    def tasks(self):
        return self.envs

    @tasks.setter
    def tasks(self, envs):
        self.envs = envs

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        self._check_steps_bound()
        obs, reward, done, truncated, info = self.get_active_env().step(action)
        info["seq_idx"] = self.cur_seq_idx

        self.cur_step += 1
        if self.cur_step % self.steps_per_env == 0:
            # If we hit limit for current env, end the episode.
            # This may cause border episodes to end before self-terminating.
            done = True

            if self.cur_seq_idx < self.num_tasks - 1:
                self.cur_seq_idx += 1
                # Lazily initialize next environment
                self._initialize_env(self.cur_seq_idx)

        return obs, reward, done, truncated, info

    def render(self, mode="rgb_array"):
        self.get_active_env().render(mode)

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None, ) -> Tuple[Union[ndarray, ndarray],
                                                                                              Dict[str, Any]]:
        self._check_steps_bound()
        return self.get_active_env().reset()

    def get_statistics(self, mode: str = '') -> Dict[str, float]:
        return self.get_active_env().get_statistics(mode)

    def clear_episode_statistics(self) -> None:
        return self.get_active_env().clear_episode_statistics()
