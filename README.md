# COOM - Super Mario Bros Edition

COOM (originally Continual Doom) has been transformed into a Continual Reinforcement Learning benchmark based on **Super Mario Bros**. This benchmark consists of task sequences across the 8 worlds of Super Mario Bros, with 4 stages per world, designed for task-incremental continual learning.

> **Note:** This is a modified version of COOM that replaces ViZDoom with stable-retro for Super Mario Bros environments.

<p align="center">
  <img src="assets/gifs/demo1.gif" alt="Demo1" style="vertical-align: top;"/>
  <img src="assets/gifs/demo2.gif" alt="Demo2" style="vertical-align: top;"/>
</p>

## Installation

### Prerequisites
- Python 3.8+
- A Super Mario Bros ROM file (you must legally own this)

### Install from source:

1. Clone the repository
```bash
$ git clone https://github.com/hyintell/COOM
```

2. Navigate into the repository
```bash
$ cd COOM
```

3. Install stable-retro (required for Mario environments)
```bash
$ pip install stable-retro
```

4. Install COOM from source
```bash
$ pip install -e .
```

5. Set up the Mario ROM integration
```bash
# The mario.stimuli folder is already included in the repository
# Make sure you have the Super Mario Bros ROM file and import it using retro
$ python -m retro.import /path/to/your/roms/
```

## Worlds and Levels

The benchmark contains **8 worlds** from Super Mario Bros, each with **4 stages**:

| World | Stages | Description | Difficulty |
|-------|--------|-------------|------------|
| **World 1** | 1-1, 1-2, 1-3, 1-4 | Classic overground, underground, tree-top, and castle | Easy |
| **World 2** | 2-1, 2-2, 2-3, 2-4 | Water world with swimming mechanics | Medium |
| **World 3** | 3-1, 3-2, 3-3, 3-4 | Night levels with aggressive enemies | Medium |
| **World 4** | 4-1, 4-2, 4-3, 4-4 | Island platforms with water gaps | Medium-Hard |
| **World 5** | 5-1, 5-2, 5-3, 5-4 | Cloud world with vertical platforming | Hard |
| **World 6** | 6-1, 6-2, 6-3, 6-4 | Ice world with slippery physics | Hard |
| **World 7** | 7-1, 7-2, 7-3, 7-4 | Pipe world with Piranha Plants | Very Hard |
| **World 8** | 8-1, 8-2, 8-3, 8-4 | Final world with all enemy types | Extreme |

Each level is accessible via its name (e.g., `'Level1-1'`, `'Level3-2'`, `'Level8-4'`).

## Reward Structure

The Mario benchmark uses a **progress-focused reward** structure:
- **Primary Reward**: Horizontal movement (x-position progress through the level)
- **Time Penalty**: Small negative reward per frame (encourages faster completion)
- **Success Metric**: Maximum x-position reached (normalized between lower and upper bounds)

### Reward Options
- **Dense Rewards**: Position progress + time penalty
- **Sparse Rewards**: Position progress only

## Action Space

The action space is based on the NES controller with 12 discrete actions:

| Movement | Jump (A) | Run/Fire (B) | Action Description |
|----------|----------|--------------|-------------------|
| None | No | No | Stand still |
| None | No | Yes | Run in place / Fire |
| None | Yes | No | Jump in place |
| None | Yes | Yes | High jump in place |
| Left | No | No | Walk left |
| Left | No | Yes | Run left |
| Left | Yes | No | Jump left |
| Left | Yes | Yes | Run-jump left |
| Right | No | No | Walk right |
| Right | No | Yes | Run right |
| Right | Yes | No | Jump right |
| Right | Yes | Yes | Run-jump right |

## Continual Learning Sequences

The benchmark includes **6 predefined continual learning sequences**:

### 1. WORLD_PROGRESSION_4
Progress through the first 4 worlds, using stage 1 of each:
- Tasks: Level1-1 → Level2-1 → Level3-1 → Level4-1
- **Purpose**: Gradual difficulty increase across different world themes

### 2. WORLD_PROGRESSION_8
Progress through all 8 worlds, using stage 1 of each:
- Tasks: Level1-1 → Level2-1 → ... → Level8-1
- **Purpose**: Maximum variety in visual themes and mechanics

### 3. STAGE_TYPES_4
Experience all 4 stage types (X-1, X-2, X-3, X-4) across different worlds:
- Tasks: Level1-1 → Level2-2 → Level3-3 → Level4-4
- **Purpose**: Learn different stage patterns (overground, underground, athletic, castle)

### 4. WORLD_COMPLETE
Complete all 4 stages of World 1, then all 4 stages of World 4:
- Tasks: Level1-1 → Level1-2 → Level1-3 → Level1-4 → Level4-1 → Level4-2 → Level4-3 → Level4-4
- **Purpose**: Master individual worlds completely before moving on

### 5. DIFFICULTY_CURVE
Progress through increasing difficulty:
- Tasks: Level1-1 → Level2-1 → ... → Level8-1
- **Purpose**: Systematic difficulty progression

### 6. MIXED_WORLDS
Alternate between different worlds and stages:
- Tasks: Level1-1 → Level3-2 → Level5-3 → Level7-4 → Level2-1 → Level4-2 → Level6-3 → Level8-4
- **Purpose**: Test adaptability with frequent context switches

## Quick Start

### Run a Single Level

```python
from COOM.env.builder import make_env
from COOM.utils.config import Scenario

# Create a World 1 environment running Level 1-1
env = make_env(Scenario.WORLD1, task='Level1-1')

obs, info = env.reset()
for _ in range(1000):
    action = env.action_space.sample()  # Random agent
    obs, reward, done, truncated, info = env.step(action)
    if done or truncated:
        break

env.close()
```

### Run a Continual Learning Sequence

```python
from COOM.env.continual import ContinualLearningEnv
from COOM.utils.config import Sequence

# Create a continual learning environment
cl_env = ContinualLearningEnv(
    sequence=Sequence.WORLD_PROGRESSION_8,
    steps_per_env=10000
)

obs, info = cl_env.reset()
for _ in range(80000):  # 8 worlds × 10000 steps
    action = cl_env.action_space.sample()
    obs, reward, done, truncated, info = cl_env.step(action)
    if done or truncated:
        obs, info = cl_env.reset()

cl_env.close()
```

### Run Example Scripts

Test a single world:
```bash
$ python COOM/examples/run_single.py --scenario world1 --task Level1-1 --render
```

Test a continual learning sequence:
```bash
$ python COOM/examples/run_sequence.py --sequence WORLD_PROGRESSION_4 --steps-per-env 1000 --render
```

## Custom Integration Path

The Mario levels use a custom stable-retro integration located in `mario.stimuli/`. This integration includes:
- **data.json**: Memory addresses for game state variables (score, coins, position, lives, etc.)
- **scenario.json**: Reward and episode termination conditions
- **metadata.json**: Benchmark performance metadata
- **Level states**: Pre-generated save states for all 32 levels (8 worlds × 4 stages)

## Game State Variables

The following game variables are accessible for reward shaping and statistics:

| Variable | Description | Type |
|----------|-------------|------|
| `score` | Game score | int |
| `coins` | Coins collected | int |
| `lives` | Remaining lives | int |
| `xscrollLo` / `xscrollHi` | Horizontal scroll position (combined for full x-pos) | int |
| `player_x_posLo` / `player_x_posHi` | Player X position | int |
| `player_y_pos` | Player Y position | int |
| `time` | Time remaining | int |
| `world` | Current world | int |
| `stage` | Current stage | int |

## Architecture Overview

```
COOM/
├── env/
│   ├── scenario.py          # MarioEnv base class (replaces DoomEnv)
│   ├── continual.py         # ContinualLearningEnv wrapper
│   ├── builder.py           # Environment factory functions
│   └── scenarios/
│       ├── world1/          # World 1 scenario (Levels 1-1 to 1-4)
│       ├── world2/          # World 2 scenario
│       ├── ...
│       └── world8/          # World 8 scenario
├── wrappers/
│   ├── reward.py            # Reward shaping wrappers (adapted for Mario)
│   └── observation.py       # Observation preprocessing
├── utils/
│   └── config.py            # Scenario and sequence definitions
└── mario.stimuli/           # Custom retro integration
    ├── SuperMarioBros-Nes/
    │   ├── data.json        # Game state variable addresses
    │   ├── scenario.json    # Reward/done conditions
    │   └── metadata.json    # Benchmark metadata
    └── generate_sublevels.py
```

## Key Differences from Original COOM

| Aspect | Original COOM | Mario COOM |
|--------|---------------|------------|
| Game Engine | ViZDoom | stable-retro |
| Game | Doom (1993) | Super Mario Bros (1985) |
| Scenarios | 8 custom scenarios | 8 worlds (32 levels total) |
| Tasks per scenario | 2-9 variations | 4 stages per world |
| Action Space | 12 actions (turn, move, shoot/jump) | 12 actions (move, jump, run) |
| Observation | 160×120 RGB | 224×256 RGB (NES resolution) |
| Success Metric | Scenario-dependent | Horizontal progress (x-position) |
| Reward | Dense/sparse per scenario | Progress-focused with time penalty |

## Training Agents

The benchmark is compatible with any RL algorithm that works with Gymnasium environments. Example training scripts using SAC and other continual learning methods can be found in the `CL/` directory.

## Citation

If you use this Mario version of COOM in your research, please cite the original COOM benchmark:

```bibtex
@inproceedings{tomilin2023coom,
  title={COOM: A Game Benchmark for Continual Reinforcement Learning},
  author={Tomilin, Tristan and Ghumare, Meng Fang and others},
  booktitle={Thirty-seventh Conference on Neural Information Processing Systems Datasets and Benchmarks Track},
  year={2023}
}
```

## License

MIT License - see LICENSE file for details.

## Acknowledgments

- Original COOM benchmark by Tristan Tomilin et al.
- Super Mario Bros © Nintendo
- stable-retro by OpenAI (maintained by the community)
