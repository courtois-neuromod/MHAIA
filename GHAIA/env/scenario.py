from collections import deque
from pathlib import Path
from typing import Dict, Tuple, Any, List, Optional, Callable
import os

import cv2
import gymnasium
import numpy as np
import retro

from COOM.env.base import BaseEnv


class MarioEnv(BaseEnv):
    """
    A foundational class for creating Super Mario Bros-based environments for reinforcement learning.

    This class manages the core functionality required for Mario-based environments,
    including game initialization, state management, and rendering using stable-retro.

    Attributes:
        env_name (str): Identifier for the specific Mario level (e.g., 'Level1-1').
        task_idx (int): Index of the current task (stage within world).
        scenario (str): Name of the scenario module (world number).
        n_tasks (int): Total number of tasks within the environment (4 stages per world).
        frame_skip (int): Number of frames to skip for each action.
        record_every (int): Frequency of recording episodes.
        metadata (Dict): Metadata for the environment, including supported render modes.
        viewer (Any): Viewer instance for rendering (if applicable).
        env (retro.RetroEnv): Instance of the stable-retro environment.
        game_res (Tuple[int, int, int]): Resolution of the game screen (224x256x3).
        _action_space (gymnasium.spaces.Discrete): The action space of the environment.
        _observation_space (gymnasium.spaces.Box): The observation space of the environment.
        user_variables (Dict[str, float]): Custom variables for tracking game state.
        game_variable_buffer (deque): A buffer for storing recent game variables for statistics.
        prev_state_data (Dict): Previous frame's game state for delta calculations.

    Args:
        env (str): Name of the specific Mario level (e.g., 'Level1-1').
        action_space_fn (Callable): Function to generate the action space.
        task_idx (int): Index of the current task.
        num_tasks (int): Total number of tasks (typically 4 stages per world).
        frame_skip (int): Number of frames to skip for each action.
        record_every (int): Frequency of recording episodes.
        seed (int): Seed for random number generators.
        render (bool): Whether to enable rendering.
        render_sleep (float): Time to sleep between rendering frames.
        test_only (bool): Whether the environment is being used for testing only.
        variable_queue_length (int): Length of the game variable buffer.
    """

    def __init__(self,
                 env: str = 'Level1-1',
                 action_space_fn: Callable = None,
                 task_idx: int = 0,
                 num_tasks: int = 4,
                 frame_skip: int = 4,
                 record_every: int = 100,
                 seed: int = 0,
                 render: bool = True,
                 render_sleep: float = 0.0,
                 test_only: bool = False,
                 variable_queue_length: int = 5):
        super().__init__()
        self.env_name = env
        self.task_idx = task_idx
        self.scenario = self.__module__.split('.')[-1]
        self.n_tasks = num_tasks
        self.frame_skip = frame_skip

        # Recording
        self.metadata['render.modes'] = 'rgb_array'
        self.record_every = record_every
        self.viewer = None
        self.render_sleep = render_sleep
        self.render_enabled = render

        # Set up the custom integration path for Mario
        integration_path = str(Path(__file__).parent.parent.parent.resolve() / 'mario.stimuli')
        retro.data.Integrations.add_custom_path(integration_path)

        # Initialize the retro environment
        self.game = retro.make(
            game='SuperMarioBros-Nes',
            state=env,
            inttype=retro.data.Integrations.CUSTOM,
            render_mode='rgb_array' if render or test_only else None
        )

        # Set seed
        self.game.action_space.seed(seed)
        self.game.observation_space.seed(seed)

        # Define the observation space (retro NES resolution is 224x256x3)
        self.game_res = self.game.observation_space.shape
        self._observation_space = self.game.observation_space

        # Define the action space
        self.available_actions = action_space_fn()
        self._action_space = gymnasium.spaces.Discrete(len(self.available_actions))

        # Initialize the user variable dictionary
        self.user_variables = {var: 0.0 for var in self.user_vars}

        # Initialize the game variable buffer
        self.game_variable_buffer = deque(maxlen=variable_queue_length)

        # Track previous state data for delta calculations
        self.prev_state_data = {}

    @property
    def task(self) -> str:
        return self.env_name

    @property
    def name(self) -> str:
        return f'{self.scenario}-{self.env_name}'

    @property
    def task_id(self):
        return self.task_idx

    @property
    def num_tasks(self) -> int:
        return self.n_tasks

    @property
    def user_vars(self) -> List[str]:
        """
        Returns the list of user-defined variable names to track.
        Override in subclasses to specify which game variables to track.
        """
        return []

    @property
    def performance_upper_bound(self) -> float:
        raise NotImplementedError

    @property
    def performance_lower_bound(self) -> float:
        raise NotImplementedError

    @property
    def action_space(self) -> gymnasium.spaces.Discrete:
        return self._action_space

    @property
    def observation_space(self) -> gymnasium.Space:
        return self._observation_space

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[
        np.ndarray, Dict[str, Any]]:
        """
        Resets the environment to its initial state and returns the initial observation.

        Args:
            seed (Optional[int]): Seed for random number generator.
            options (Optional[dict]): Additional options for environment reset.

        Returns:
            observation (np.ndarray): Initial state observation of the environment.
            info (Dict[str, Any]): Additional information about the initial state.
        """
        if seed is not None:
            self.game.action_space.seed(seed)
            self.game.observation_space.seed(seed)

        observation, info = self.game.reset(seed=seed, options=options)
        self.clear_episode_statistics()

        # Initialize previous state data
        self.prev_state_data = self._get_state_dict()

        return observation, info

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Perform an action in the environment and observe the result.

        Args:
            action (int): An action provided by the agent.

        Returns:
            observation (np.ndarray): The current state observation after taking the action.
            reward (float): The reward achieved by the action.
            done (bool): Whether the episode has ended.
            truncated (bool): Whether the episode was truncated.
            info (Dict[str, Any]): Additional information about the environment and episode.
        """
        # Convert discrete action index to button array
        button_action = self.available_actions[action]

        # Execute action with frame skip
        total_reward = 0.0
        for _ in range(self.frame_skip):
            observation, reward, done, truncated, info = self.game.step(button_action)
            total_reward += reward
            if done or truncated:
                break

        # Store current game state variables
        current_state_data = self._get_state_dict()
        self.game_variable_buffer.append(current_state_data)

        if self.render_enabled:
            self.render('human')

        # Update statistics
        self.store_statistics(self.game_variable_buffer)

        # Update previous state for next iteration
        self.prev_state_data = current_state_data

        return observation, total_reward, done, truncated, info

    def _get_state_dict(self) -> Dict[str, float]:
        """
        Convert GameData to a dictionary of all tracked variables.

        Returns:
            Dictionary mapping variable names to their current values.
        """
        state_dict = {}
        # List of all variables we track from data.json
        var_names = [
            'area', 'coins', 'enemy_drawn15', 'enemy_drawn16', 'enemy_drawn17',
            'enemy_drawn18', 'enemy_drawn19', 'enemy_kill30', 'enemy_kill31',
            'enemy_kill32', 'enemy_kill33', 'enemy_kill34', 'enemy_kill35',
            'fireball_counter', 'jump_airborne', 'levelHi', 'levelLo',
            'level_layout', 'lives', 'moving_direction', 'player_sprite',
            'player_state', 'player_x_posHi', 'player_x_posLo', 'player_y_pos',
            'player_y_screen', 'powerstate', 'powerup_appear', 'powerup_yes_no',
            'score', 'scrolling', 'stage', 'star_timer', 'time',
            'walk_animation', 'world', 'xscrollHi', 'xscrollLo'
        ]

        for var_name in var_names:
            try:
                state_dict[var_name] = float(self.game.data.lookup_value(var_name))
            except:
                state_dict[var_name] = 0.0

        return state_dict

    def get_state_variable(self, var_name: str) -> float:
        """
        Retrieves a game state variable value.

        Args:
            var_name (str): The name of the game variable (e.g., 'score', 'coins', 'xscrollLo').

        Returns:
            value (float): The current value of the specified game variable.
        """
        try:
            return float(self.game.data.lookup_value(var_name))
        except:
            return 0.0

    def get_state_variable_delta(self, var_name: str) -> float:
        """
        Retrieves the change in a game state variable since the last step.

        Args:
            var_name (str): The name of the game variable.

        Returns:
            delta (float): The change in the variable value.
        """
        current = self.get_state_variable(var_name)
        previous = float(self.prev_state_data.get(var_name, 0))
        return current - previous

    def get_statistics(self, mode: str = '') -> Dict[str, float]:
        """
        Retrieves statistics about the environment's performance or state.

        Args:
            mode (str): A specifier for the type of statistics to retrieve.

        Returns:
            metrics (Dict[str, float]): A dictionary containing statistical data.
        """
        metrics = self.extra_statistics(mode)
        metrics[f'{mode}/success'] = self.get_success()
        return metrics

    def get_success(self) -> float:
        """
        Calculates and returns a normalized success metric between [0, 1] for the current environment state.

        Returns:
            success_norm (float): Normalized success metric.
        """
        success_norm = (self.get_success_metric() - self.performance_lower_bound) / (
                self.performance_upper_bound - self.performance_lower_bound)
        return float(np.clip(success_norm, 0.0, 1.0))

    def extra_statistics(self, mode: str = '') -> Dict[str, float]:
        """
        Retrieves additional statistics specific to the scenario. Mostly game variables.

        Args:
            mode (str): A specifier to distinguish which environment the statistic is for (train/test).

        Returns:
            statistics (Dict[str, float]): A dictionary containing additional statistical data.
        """
        return {}

    def store_statistics(self, game_vars: deque) -> None:
        """
        Stores statistics based on the game variables.

        Args:
            game_vars (deque): A deque containing game variables for statistics.
        """
        pass

    def get_success_metric(self) -> float:
        """
        Retrieves the success metric based on the current state of the environment.

        Returns:
            success_metric (float): The calculated success metric.
        """
        raise NotImplementedError

    def reward_wrappers_dense(self) -> List[gymnasium.RewardWrapper]:
        """
        Returns a list of reward wrapper classes for the dense reward setting.

        Returns:
            List[gymnasium.RewardWrapper]: A list of reward wrapper classes.
        """
        raise NotImplementedError

    def reward_wrappers_sparse(self) -> List[gymnasium.RewardWrapper]:
        """
        Returns a list of reward wrapper classes for the sparse reward setting.

        Returns:
            List[gymnasium.RewardWrapper]: A list of reward wrapper classes.
        """
        raise NotImplementedError

    def get_and_update_user_var(self, var_name: str) -> float:
        """
        Retrieves and updates a user-defined variable from the game.

        Args:
            var_name (str): The name of the game variable to retrieve and update.

        Returns:
            prev_var (float): The previous value of the specified game variable.
        """
        prev_var = self.user_variables[var_name]
        self.user_variables[var_name] = self.get_state_variable(var_name)
        return prev_var

    def render(self, mode="rgb_array"):
        """
        Renders the current state of the environment based on the specified mode.

        Args:
            mode (str): The mode for rendering (e.g., 'human', 'rgb_array').

        Returns:
            img (List[np.ndarray] or np.ndarray): Rendered image of the environment state.
        """
        img = self.game.render()
        if img is None:
            img = np.zeros(self.game_res, dtype=np.uint8)

        if mode == 'human':
            if not self.render_enabled:
                return [img]
            try:
                # Render the image to the screen with swapped red and blue channels
                cv2.imshow('Super Mario Bros', img[:, :, [2, 1, 0]])
                cv2.waitKey(1)
            except Exception as e:
                print(f'Screen rendering unsuccessful: {e}')
                return np.zeros(img.shape)
        return [img]

    def video_schedule(self, episode_id):
        """
        Determines whether a video of the current episode should be recorded.

        Args:
            episode_id (int): The identifier of the current episode.

        Returns:
            bool: True if the episode should be recorded, False otherwise.
        """
        return not episode_id % self.record_every

    def clear_episode_statistics(self) -> None:
        """
        Clears or resets statistics collected during an episode.
        """
        self.user_variables = {var: 0.0 for var in self.user_vars}
        self.game_variable_buffer.clear()
        self.prev_state_data = {}

    def close(self):
        """
        Performs cleanup and closes the environment.
        """
        self.game.close()

    def get_active_env(self):
        """
        Returns the currently active environment.

        Returns:
            The active environment instance.
        """
        return self
