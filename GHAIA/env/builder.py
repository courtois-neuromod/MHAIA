import itertools
from typing import Dict, List
import numpy as np

from gymnasium.wrappers import NormalizeObservation, FrameStack, RecordVideo

from GHAIA.env.scenario import MarioEnv
from GHAIA.utils.config import Sequence, sequence_scenarios, sequence_tasks, scenario_config, Scenario, \
    default_wrapper_config
from GHAIA.wrappers.observation import Augment, Resize, Rescale, RGBStack


def make_sequence(sequence: Sequence,
                  random_order: bool = False,
                  scenarios_kwargs: List[Dict[str, any]] = None,
                  mario_kwargs: Dict[str, any] = None,
                  wrapper_config: Dict[str, any] = None,
                  task_idx: int = None) -> List[MarioEnv]:
    """
    Creates a list of Mario environments based on the given sequence configuration.

    Args:
        sequence (Sequence): The sequence enumeration to determine which scenarios to include.
        random_order (bool): Whether to randomize the order of the scenarios.
        task_idx (int): Optional task index to be used for all environments.
        scenarios_kwargs (List[Dict[str, any]]): List of dictionaries with specific kwargs for each scenario.
        mario_kwargs (Dict[str, any]): Common kwargs applicable to all Mario environments.
        wrapper_config (Dict[str, any]): Configuration for environment wrappers.

    Returns:
        List[MarioEnv]: A list of Mario environment instances.
    """

    # Retrieve scenarios and tasks based on the sequence
    scenarios = sequence_scenarios[sequence]
    tasks = sequence_tasks[sequence]
    return make_envs(scenarios, tasks, random_order, task_idx, scenarios_kwargs, mario_kwargs, wrapper_config)


def make_envs(scenarios: List[Scenario],
              tasks: List[str],
              random_order: bool = False,
              task_idx: int = None,
              scenarios_kwargs: List[Dict[str, any]] = None,
              mario_kwargs: Dict[str, any] = None,
              wrapper_config: Dict[str, any] = None) -> List[MarioEnv]:
    """
    Creates multiple Mario environments from scenarios and tasks.

    Args:
        scenarios: List of Scenario enums
        tasks: List of task/level names
        random_order: Whether to shuffle scenarios and tasks
        task_idx: Optional task index override
        scenarios_kwargs: Per-scenario keyword arguments
        mario_kwargs: Common Mario environment arguments
        wrapper_config: Wrapper configuration

    Returns:
        List of created MarioEnv instances
    """

    # Optionally shuffle scenarios and tasks for randomization
    if random_order:
        import random
        random.shuffle(scenarios)
        random.shuffle(tasks)

    # Default kwargs for scenarios and Mario environments
    scenarios_kwargs = scenarios_kwargs or [{} for _ in range(len(scenarios))]
    mario_kwargs = mario_kwargs or {}

    # Create and wrap environments
    envs = []
    for i, pair in enumerate(itertools.product(zip(scenarios, scenarios_kwargs), tasks)):
        # If task_idx is specified, use that otherwise use the current index.
        task_id = task_idx if task_idx is not None else i
        scenario_and_kwargs = pair[0]
        task = pair[1]
        scenario = scenario_and_kwargs[0]
        scenario_kwargs = scenario_and_kwargs[1]
        env = make_env(scenario, task, task_id, scenario_kwargs, mario_kwargs, wrapper_config)
        envs.append(env)
    return envs


def make_env(scenario: Scenario,
             task: str = 'Level1-1',
             task_idx: int = 0,
             scenario_kwargs: Dict[str, any] = None,
             mario_kwargs: Dict[str, any] = None,
             wrapper_config: Dict[str, any] = None) -> MarioEnv:
    """
    Creates a single Mario environment instance with specified configurations.

    Args:
        scenario (Scenario): The specific Mario world/scenario to create.
        task (str): The task/level name within the scenario (e.g., 'Level1-1').
        task_idx (int): The index of the task within the scenario.
        scenario_kwargs (Dict[str, any]): Additional kwargs for the scenario.
        mario_kwargs (Dict[str, any]): Common kwargs for Mario environments.
        wrapper_config (Dict[str, any]): Configuration for environment wrappers.

    Returns:
        MarioEnv: An instance of the Mario environment.
    """

    # Retrieve the scenario class and create an instance
    scenario_class = scenario_config[scenario]['class']
    scenario_kwargs = scenario_kwargs or {}

    # Build default mario_kwargs and merge with provided ones
    default_mario_kwargs = {
        'env': task,
        'task_idx': task_idx,
        'action_space_fn': build_mario_actions
    }

    # Merge provided mario_kwargs with defaults (provided values take precedence)
    if mario_kwargs:
        default_mario_kwargs.update(mario_kwargs)
    mario_kwargs = default_mario_kwargs

    env = scenario_class(mario_kwargs, **scenario_kwargs)

    # Apply wrappers to the environment
    env = wrap_env(env, wrapper_config or default_wrapper_config)
    return env


def wrap_env(env: MarioEnv, wrap_conf: Dict[str, any]):
    """
    Applies a series of wrappers to the Mario environment based on the provided configuration.

    Args:
        env (MarioEnv): The Mario environment to be wrapped.
        wrap_conf (Dict[str, any]): Configuration dict specifying which wrappers to apply.

    Returns:
        gymnasium.Env: The wrapped environment.
    """

    # Apply reward wrappers based on the sparse_rewards configuration
    sparse_rewards = wrap_conf.get('sparse_rewards', False)
    reward_wrappers = env.reward_wrappers_sparse() if sparse_rewards else env.reward_wrappers_dense()
    for wrapper in reward_wrappers:
        env = wrapper.wrapper_class(env, **wrapper.kwargs)

    # Apply various observation and utility wrappers
    if wrap_conf.get('augment', False):
        env = Augment(env, wrap_conf['augmentation'])
    if wrap_conf.get('resize', False):
        assert wrap_conf.get('frame_height', None) is not None and wrap_conf.get('frame_width', None) is not None
        env = Resize(env, wrap_conf['frame_height'], wrap_conf['frame_width'])
    if wrap_conf.get('rescale', False):
        env = Rescale(env)
    if wrap_conf.get('normalize_observation', False):
        env = NormalizeObservation(env)
    if wrap_conf.get('frame_stack', False):
        env = FrameStack(env, wrap_conf['frame_stack'])
    if wrap_conf.get('lstm', False):
        env = RGBStack(env)
    if wrap_conf.get('record', False):
        env = RecordVideo(env, wrap_conf['record_dir'], episode_trigger=env.video_schedule, name_prefix=f'{env.name}')
    return env


def build_mario_actions():
    """
    Builds action space for Super Mario Bros using NES controller buttons.

    NES Controller buttons (indices for stable-retro):
    - Button indices: [B, None, Select, Start, Up, Down, Left, Right, A]
    - For Mario, we use: [B(run/fire), _, _, _, Up, Down, Left, Right, A(jump)]

    This creates a simplified action space with 12 actions:
    - Movement: Left, Right, or None
    - Jump: A button or not
    - Run/Fire: B button or not

    Returns:
        List of action arrays (each is a 9-element boolean array for NES buttons)
    """
    actions = []

    # Movement options: Left, Right, None
    movement_options = [
        [0, 0, 0, 0, 0, 0, 0, 0, 0],  # No movement
        [0, 0, 0, 0, 0, 0, 1, 0, 0],  # Left
        [0, 0, 0, 0, 0, 0, 0, 1, 0],  # Right
    ]

    # Jump options: Jump (A) or not
    jump_options = [
        [0, 0, 0, 0, 0, 0, 0, 0, 0],  # No jump
        [0, 0, 0, 0, 0, 0, 0, 0, 1],  # Jump (A button)
    ]

    # Run/Fire options: Run (B) or not
    run_options = [
        [0, 0, 0, 0, 0, 0, 0, 0, 0],  # No run
        [1, 0, 0, 0, 0, 0, 0, 0, 0],  # Run/Fire (B button)
    ]

    # Combine all options (3 movement × 2 jump × 2 run = 12 actions)
    for movement in movement_options:
        for jump in jump_options:
            for run in run_options:
                # Combine button presses (OR operation for each button)
                action = [
                    max(movement[i], jump[i], run[i])
                    for i in range(9)
                ]
                actions.append(np.array(action, dtype=np.uint8))

    return actions


def build_simple_mario_actions():
    """
    Builds a simplified action space for Super Mario Bros with only essential actions.

    Creates 7 core actions:
    - NOOP (do nothing)
    - RIGHT (move right)
    - RIGHT + A (jump right)
    - RIGHT + B (run right)
    - RIGHT + A + B (jump and run right)
    - LEFT (move left)
    - A (jump in place)

    Returns:
        List of action arrays for NES controller
    """
    # [B, None, Select, Start, Up, Down, Left, Right, A]
    actions = [
        [0, 0, 0, 0, 0, 0, 0, 0, 0],  # NOOP
        [0, 0, 0, 0, 0, 0, 0, 1, 0],  # RIGHT
        [0, 0, 0, 0, 0, 0, 0, 1, 1],  # RIGHT + A (jump right)
        [1, 0, 0, 0, 0, 0, 0, 1, 0],  # RIGHT + B (run right)
        [1, 0, 0, 0, 0, 0, 0, 1, 1],  # RIGHT + B + A (run and jump right)
        [0, 0, 0, 0, 0, 0, 1, 0, 0],  # LEFT
        [0, 0, 0, 0, 0, 0, 0, 0, 1],  # A (jump in place)
    ]

    return [np.array(action, dtype=np.uint8) for action in actions]


# Backward compatibility aliases
build_multi_discrete_actions = build_mario_actions
build_discrete_actions = build_simple_mario_actions
