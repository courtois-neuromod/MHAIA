from enum import Enum

from MHAIA.env.scenarios.world1.world1 import World1
from MHAIA.env.scenarios.world2.world2 import World2
from MHAIA.env.scenarios.world3.world3 import World3
from MHAIA.env.scenarios.world4.world4 import World4
from MHAIA.env.scenarios.world5.world5 import World5
from MHAIA.env.scenarios.world6.world6 import World6
from MHAIA.env.scenarios.world7.world7 import World7
from MHAIA.env.scenarios.world8.world8 import World8
from MHAIA.utils.augmentations import random_conv, random_shift, random_noise


class Augmentation(Enum):
    CONV = random_conv
    SHIFT = random_shift
    NOISE = random_noise


class Sequence(Enum):
    """
    Continual Learning sequences for Super Mario Bros.

    - WORLD_PROGRESSION_4: Progress through first 4 worlds (1-4), one stage per world
    - WORLD_PROGRESSION_8: Progress through all 8 worlds (1-8), one stage per world
    - STAGE_TYPES_4: Experience all 4 stage types across different worlds
    - WORLD_COMPLETE: Complete all 4 stages of 2 worlds
    - DIFFICULTY_CURVE: Ordered by difficulty (easy to hard levels)
    - MIXED_WORLDS: Interleaved worlds and stages
    """
    WORLD_PROGRESSION_4 = 1
    WORLD_PROGRESSION_8 = 2
    STAGE_TYPES_4 = 3
    WORLD_COMPLETE = 4
    DIFFICULTY_CURVE = 5
    MIXED_WORLDS = 6


class Scenario(Enum):
    """
    The 8 worlds of Super Mario Bros.
    Each world contains 4 stages (levels).
    """
    WORLD1 = 1
    WORLD2 = 2
    WORLD3 = 3
    WORLD4 = 4
    WORLD5 = 5
    WORLD6 = 6
    WORLD7 = 7
    WORLD8 = 8


# Configuration for each scenario (world)
scenario_config = {
    Scenario.WORLD1: {
        'class': World1,
        'args': ['reward_position', 'penalty_time', 'reward_coin']
    },
    Scenario.WORLD2: {
        'class': World2,
        'args': ['reward_position', 'penalty_time', 'reward_coin']
    },
    Scenario.WORLD3: {
        'class': World3,
        'args': ['reward_position', 'penalty_time', 'reward_coin']
    },
    Scenario.WORLD4: {
        'class': World4,
        'args': ['reward_position', 'penalty_time', 'reward_coin']
    },
    Scenario.WORLD5: {
        'class': World5,
        'args': ['reward_position', 'penalty_time', 'reward_coin']
    },
    Scenario.WORLD6: {
        'class': World6,
        'args': ['reward_position', 'penalty_time', 'reward_coin']
    },
    Scenario.WORLD7: {
        'class': World7,
        'args': ['reward_position', 'penalty_time', 'reward_coin']
    },
    Scenario.WORLD8: {
        'class': World8,
        'args': ['reward_position', 'penalty_time', 'reward_coin']
    },
}

# All worlds for easy reference
ALL_WORLDS = [Scenario.WORLD1, Scenario.WORLD2, Scenario.WORLD3, Scenario.WORLD4,
              Scenario.WORLD5, Scenario.WORLD6, Scenario.WORLD7, Scenario.WORLD8]

# Stage/level names for each world
# Each world has 4 stages: Level{W}-{S} where W=world (1-8), S=stage (1-4)
STAGE_1_LEVELS = ['Level1-1', 'Level2-1', 'Level3-1', 'Level4-1',
                  'Level5-1', 'Level6-1', 'Level7-1', 'Level8-1']
STAGE_2_LEVELS = ['Level1-2', 'Level2-2', 'Level3-2', 'Level4-2',
                  'Level5-2', 'Level6-2', 'Level7-2', 'Level8-2']
STAGE_3_LEVELS = ['Level1-3', 'Level2-3', 'Level3-3', 'Level4-3',
                  'Level5-3', 'Level6-3', 'Level7-3', 'Level8-3']
STAGE_4_LEVELS = ['Level1-4', 'Level2-4', 'Level3-4', 'Level4-4',
                  'Level5-4', 'Level6-4', 'Level7-4', 'Level8-4']

# All stages for a given world
WORLD_STAGES = {
    Scenario.WORLD1: ['Level1-1', 'Level1-2', 'Level1-3', 'Level1-4'],
    Scenario.WORLD2: ['Level2-1', 'Level2-2', 'Level2-3', 'Level2-4'],
    Scenario.WORLD3: ['Level3-1', 'Level3-2', 'Level3-3', 'Level3-4'],
    Scenario.WORLD4: ['Level4-1', 'Level4-2', 'Level4-3', 'Level4-4'],
    Scenario.WORLD5: ['Level5-1', 'Level5-2', 'Level5-3', 'Level5-4'],
    Scenario.WORLD6: ['Level6-1', 'Level6-2', 'Level6-3', 'Level6-4'],
    Scenario.WORLD7: ['Level7-1', 'Level7-2', 'Level7-3', 'Level7-4'],
    Scenario.WORLD8: ['Level8-1', 'Level8-2', 'Level8-3', 'Level8-4'],
}

# Sequence definitions
# Map each sequence type to its list of scenarios and tasks

sequence_scenarios = {
    # Progress through worlds 1-4, using first stage of each
    Sequence.WORLD_PROGRESSION_4: [Scenario.WORLD1, Scenario.WORLD2, Scenario.WORLD3, Scenario.WORLD4],

    # Progress through all 8 worlds, using first stage of each
    Sequence.WORLD_PROGRESSION_8: ALL_WORLDS,

    # Experience all 4 stage types (X-1, X-2, X-3, X-4) using different worlds
    Sequence.STAGE_TYPES_4: [Scenario.WORLD1, Scenario.WORLD2, Scenario.WORLD3, Scenario.WORLD4],

    # Complete all stages of World 1 and World 4 (2 complete worlds)
    Sequence.WORLD_COMPLETE: [Scenario.WORLD1, Scenario.WORLD1, Scenario.WORLD1, Scenario.WORLD1,
                              Scenario.WORLD4, Scenario.WORLD4, Scenario.WORLD4, Scenario.WORLD4],

    # Difficulty progression: easier levels first, harder later
    Sequence.DIFFICULTY_CURVE: [Scenario.WORLD1, Scenario.WORLD2, Scenario.WORLD3, Scenario.WORLD4,
                                Scenario.WORLD5, Scenario.WORLD6, Scenario.WORLD7, Scenario.WORLD8],

    # Mixed: alternate between different worlds and stages
    Sequence.MIXED_WORLDS: [Scenario.WORLD1, Scenario.WORLD3, Scenario.WORLD5, Scenario.WORLD7,
                            Scenario.WORLD2, Scenario.WORLD4, Scenario.WORLD6, Scenario.WORLD8],
}

sequence_tasks = {
    # World progression: use stage 1 of each world
    Sequence.WORLD_PROGRESSION_4: ['Level1-1', 'Level2-1', 'Level3-1', 'Level4-1'],

    # All 8 worlds, stage 1 each
    Sequence.WORLD_PROGRESSION_8: STAGE_1_LEVELS,

    # Experience different stage types: 1-1, 2-2, 3-3, 4-4
    Sequence.STAGE_TYPES_4: ['Level1-1', 'Level2-2', 'Level3-3', 'Level4-4'],

    # Complete World 1 and World 4 fully
    Sequence.WORLD_COMPLETE: ['Level1-1', 'Level1-2', 'Level1-3', 'Level1-4',
                              'Level4-1', 'Level4-2', 'Level4-3', 'Level4-4'],

    # Difficulty curve: progress through stages with increasing difficulty
    Sequence.DIFFICULTY_CURVE: ['Level1-1', 'Level2-1', 'Level3-1', 'Level4-1',
                                'Level5-1', 'Level6-1', 'Level7-1', 'Level8-1'],

    # Mixed worlds: alternate stage types
    Sequence.MIXED_WORLDS: ['Level1-1', 'Level3-2', 'Level5-3', 'Level7-4',
                            'Level2-1', 'Level4-2', 'Level6-3', 'Level8-4'],
}


default_wrapper_config = {
    'augment': False,
    'augmentation': 'conv',
    'resize': True,
    'frame_height': 84,
    'frame_width': 84,
    'rescale': True,
    'normalize_observation': True,
    'frame_stack': 4,
    'lstm': False,
    'record': False,
    'record_dir': 'videos',
    'sparse_rewards': False,
}
