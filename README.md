# MHAIA - Mario Human-AI Alignment Benchmark (Status: WiP)

> 🎵 Mario-hii Mario-huu Mario-hoo Mario ah haaaa! 🎵
>
> — An exhausted data manager.

**Built upon [COOM (Continual Doom)](https://github.com/TTomilin/COOM)** - This project is based on the COOM benchmark by Tristan Tomilin et al. We extend their excellent continual reinforcement learning framework to create MHAIA, a benchmark for evaluating human-AI alignment in game-based environments.

**MHAIA** (Mario Human-AI Alignment) is a continual reinforcement learning benchmark based on **Super Mario Bros**, designed to evaluate AI agents' ability to learn sequential tasks while maintaining alignment with human gameplay patterns and preferences. The benchmark consists of task sequences across the 8 worlds of Super Mario Bros, with 4 stages per world.

📄 **Paper**: [MHAIA on OpenReview](https://openreview.net/forum?id=YAVB439L9X)

<p align="center">
  <img src="assets/gifs/mario_demo1.gif" alt="Mario Demo 1" style="vertical-align: top;"/>
  <img src="assets/gifs/mario_demo2.gif" alt="Mario Demo 2" style="vertical-align: top;"/>
</p>

## Installation

### Prerequisites

- Python 3.8+
- Super Mario Bros ROM file (see ROM setup below)

### Install from source:

1. Clone the repository

```bash
git clone --recurse-submodules https://github.com/courtois-neuromod/MHAIA
```

2. Navigate into the repository

```bash
cd MHAIA
```

3. Create and activate a virtual environment

```bash
python -m venv env
source env/bin/activate  # On Windows: env\Scripts\activate
```

4. Install MHAIA with all dependencies

```bash
# For basic Mario environment usage
pip install -e .

# For continual learning experiments (includes TensorFlow, etc.)
pip install -e ".[cl]"
```

**Note**: The `[cl]` extra includes TensorFlow, TensorBoard, wandb, and other dependencies required for running continual learning experiments. If you only need to run single Mario levels, the basic installation (`pip install -e .`) is sufficient.

### ROM Setup

You need the Super Mario Bros (NES) ROM file to use MHAIA. There are two methods:

#### Method 1: Via DataLad (Recommended for CNeuromod members)

If you have access to the **CNeuromod dataset**, you can retrieve the ROM and game state files using DataLad:

```bash
# Navigate to the mario.stimuli submodule
cd mario.stimuli

# Install DataLad if not already installed
pip install datalad

# Configure your AWS credentials (provided by the CNeuromod team)
# You will need an AWS access key and secret key with access to the CNeuromod S3 bucket
export AWS_ACCESS_KEY_ID=your_key_here
export AWS_SECRET_ACCESS_KEY=your_secret_here

# Get the ROM and level state files
datalad get SuperMarioBros-Nes/
```

**Note**: Access to the CNeuromod dataset requires approval. Contact the [Courtois NeuroMod team](https://www.cneuromod.ca/) to request access.

#### Method 2: Import Your Own ROM

If you legally own Super Mario Bros, you can import your ROM file:

```bash
# Import your ROM file into stable-retro
python -m retro.import /path/to/your/roms/
```

**Required ROM specifications:**

- **Game**: Super Mario Bros (NES)
- **Region**: USA/North America (NTSC)
- **SHA-1 Hash**: `facee9c577a5262dbe33ac4930bb0b58c8c037f7`

You can verify your ROM hash with:

```bash
sha1sum /path/to/your/SuperMarioBros.nes
```

## Worlds and Levels

The benchmark contains **8 worlds** from Super Mario Bros, each with **4 stages**:

| World       | Stages             | Description                                           | Difficulty  |
| ----------- | ------------------ | ----------------------------------------------------- | ----------- |
| **World 1** | 1-1, 1-2, 1-3, 1-4 | Classic overground, underground, tree-top, and castle | Easy        |
| **World 2** | 2-1, 2-2, 2-3, 2-4 | Water world with swimming mechanics                   | Medium      |
| **World 3** | 3-1, 3-2, 3-3, 3-4 | Night levels with aggressive enemies                  | Medium      |
| **World 4** | 4-1, 4-2, 4-3, 4-4 | Island platforms with water gaps                      | Medium-Hard |
| **World 5** | 5-1, 5-2, 5-3, 5-4 | Cloud world with vertical platforming                 | Hard        |
| **World 6** | 6-1, 6-2, 6-3, 6-4 | Ice world with slippery physics                       | Hard        |
| **World 7** | 7-1, 7-2, 7-3, 7-4 | Pipe world with Piranha Plants                        | Very Hard   |
| **World 8** | 8-1, 8-2, 8-3, 8-4 | Final world with all enemy types                      | Extreme     |

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

| Movement | Jump (A) | Run/Fire (B) | Action Description  |
| -------- | -------- | ------------ | ------------------- |
| None     | No       | No           | Stand still         |
| None     | No       | Yes          | Run in place / Fire |
| None     | Yes      | No           | Jump in place       |
| None     | Yes      | Yes          | High jump in place  |
| Left     | No       | No           | Walk left           |
| Left     | No       | Yes          | Run left            |
| Left     | Yes      | No           | Jump left           |
| Left     | Yes      | Yes          | Run-jump left       |
| Right    | No       | No           | Walk right          |
| Right    | No       | Yes          | Run right           |
| Right    | Yes      | No           | Jump right          |
| Right    | Yes      | Yes          | Run-jump right      |

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
from MHAIA.env.builder import make_env
from MHAIA.utils.config import Scenario

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
from MHAIA.env.continual import ContinualLearningEnv
from MHAIA.utils.config import Sequence

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
python MHAIA/examples/run_single.py --scenario world1 --task Level1-1 --render
```

Test a continual learning sequence:

```bash
python MHAIA/examples/run_sequence.py --sequence WORLD_PROGRESSION_4 --steps-per-env 1000 --render
```

## Custom Integration Path

The Mario levels use a custom stable-retro integration located in `mario.stimuli/`. This integration includes:

- **data.json**: Memory addresses for game state variables (score, coins, position, lives, etc.)
- **scenario.json**: Reward and episode termination conditions
- **metadata.json**: Benchmark performance metadata
- **Level states**: Pre-generated save states for all 32 levels (8 worlds × 4 stages)

## Game State Variables

The following game variables are accessible for reward shaping and statistics:

| Variable                            | Description                                          | Type |
| ----------------------------------- | ---------------------------------------------------- | ---- |
| `score`                             | Game score                                           | int  |
| `coins`                             | Coins collected                                      | int  |
| `lives`                             | Remaining lives                                      | int  |
| `xscrollLo` / `xscrollHi`           | Horizontal scroll position (combined for full x-pos) | int  |
| `player_x_posLo` / `player_x_posHi` | Player X position                                    | int  |
| `player_y_pos`                      | Player Y position                                    | int  |
| `time`                              | Time remaining                                       | int  |
| `world`                             | Current world                                        | int  |
| `stage`                             | Current stage                                        | int  |

## Architecture Overview

```
MHAIA/
├── env/
│   ├── scenario.py          # MarioEnv base class
│   ├── continual.py         # ContinualLearningEnv wrapper
│   ├── builder.py           # Environment factory functions
│   └── scenarios/
│       ├── world1/          # World 1 scenario (Levels 1-1 to 1-4)
│       ├── world2/          # World 2 scenario
│       ├── ...
│       └── world8/          # World 8 scenario
├── wrappers/
│   ├── reward.py            # Reward shaping wrappers
│   └── observation.py       # Observation preprocessing
├── utils/
│   └── config.py            # Scenario and sequence definitions
└── mario.stimuli/           # Custom retro integration (submodule)
    ├── SuperMarioBros-Nes/
    │   ├── data.json        # Game state variable addresses
    │   ├── scenario.json    # Reward/done conditions
    │   └── metadata.json    # Benchmark metadata
    └── generate_sublevels.py
```

## Key Differences from Original COOM

| Aspect             | Original COOM (Doom)                | MHAIA (Mario)                      |
| ------------------ | ----------------------------------- | ---------------------------------- |
| Game Engine        | ViZDoom                             | stable-retro                       |
| Game               | Doom (1993)                         | Super Mario Bros (1985)            |
| Scenarios          | 8 custom scenarios                  | 8 worlds (32 levels total)         |
| Tasks per scenario | 2-9 variations                      | 4 stages per world                 |
| Action Space       | 12 actions (turn, move, shoot/jump) | 12 actions (move, jump, run)       |
| Observation        | 160×120 RGB                         | 224×256 RGB (NES resolution)       |
| Success Metric     | Scenario-dependent                  | Horizontal progress (x-position)   |
| Reward             | Dense/sparse per scenario           | Progress-focused with time penalty |
| Focus              | Continual RL                        | Human-AI Alignment + Continual RL  |

## Training Agents

The benchmark is compatible with any RL algorithm that works with Gymnasium environments. Example training scripts using SAC and other continual learning methods can be found in the `CL/` directory.

### Running Continual Learning Experiments

After installing with `pip install -e ".[cl]"`, you can run CL experiments:

```bash
# Activate the environment
source env/bin/activate

# Simple fine-tuning baseline (no CL method)
python -m CL.run_cl --sequence WORLD_PROGRESSION_4 \
  --steps_per_env 50000 \
  --seed 0 \
  --multihead_archs False

# With a CL method (e.g., L2 regularization)
python -m CL.run_cl --sequence WORLD_PROGRESSION_8 \
  --cl_method l2 \
  --cl_reg_coef 100 \
  --steps_per_env 50000 \
  --seed 42 \
  --multihead_archs False

# With PackNet
python -m CL.run_cl --sequence WORLD_PROGRESSION_4 \
  --cl_method packnet \
  --packnet_retrain_steps 10000 \
  --steps_per_env 100000 \
  --seed 0 \
  --multihead_archs False
```

**Available CL Methods:**

- `l2` - L2 weight regularization
- `ewc` - Elastic Weight Consolidation
- `mas` - Memory Aware Synapses
- `packnet` - PackNet (network pruning)
- `agem` - Averaged Gradient Episodic Memory
- `vcl` - Variational Continual Learning
- `owl` - Online Weighted Laplacian
- `clonex` - ClonEx

**Important Notes:**

- Use `--multihead_archs False` to avoid architecture dimension mismatches
- Results are logged to `./logs/` directory
- Add `--with_wandb` for Weights & Biases logging
- See `python -m CL.run_cl --help` for all options

For detailed CL configurations and reproducing paper results, see the [CL README](CL/README.md).

## Citation

If you use MHAIA in your research, please cite both the MHAIA benchmark and the original COOM framework it builds upon:

```bibtex
@inproceedings{harel2025human,
  title={Human-AI Alignment of Learning Trajectories in Video Games: a continual RL benchmark proposal},
  author={Harel*, Yann and Bellec*, Lune P and Paugam, Fran{\c{c}}ois and Delhaye, Hugo and Durand, Audrey},
  booktitle={Reinforcement Learning and Video Games Workshop@ RLC 2025}
}


@article{tomilin2023coom,
  title={COOM: A game benchmark for continual reinforcement learning},
  author={Tomilin, Tristan and Fang, Meng and Zhang, Yudi and Pechenizkiy, Mykola},
  journal={Advances in Neural Information Processing Systems},
  volume={36},
  pages={67794--67832},
  year={2023}
}
```

## License

MIT License - see LICENSE file for details.

## Acknowledgments

- **COOM Framework**: This project builds upon the excellent [COOM (Continual Doom)](https://github.com/TTomilin/COOM) benchmark by Tristan Tomilin et al., which provided the foundational continual reinforcement learning architecture
- Super Mario Bros © Nintendo
- stable-retro by OpenAI (maintained by the community)
- Courtois NeuroMod Project for supporting this research