# Migration Guide: DOOM COOM → Super Mario Bros COOM

This document details all changes made to transform COOM from a DOOM-based continual learning benchmark to a Super Mario Bros-based benchmark using stable-retro.

## Table of Contents
- [Overview](#overview)
- [Core Architecture Changes](#core-architecture-changes)
- [File-by-File Changes](#file-by-file-changes)
- [API Changes](#api-changes)
- [Configuration Changes](#configuration-changes)
- [Dependencies](#dependencies)
- [Key Differences](#key-differences)
- [Migration Checklist](#migration-checklist)

---

## Overview

### What Changed
- **Game Engine**: ViZDoom → stable-retro
- **Game**: DOOM (1993) → Super Mario Bros (1985, NES)
- **Scenarios**: 8 custom DOOM scenarios → 8 Mario worlds (32 levels total)
- **Focus**: Diverse objectives (shooting, survival, navigation) → Platforming progression

### Why This Works
The COOM architecture was designed to be modular and game-agnostic:
- Abstract base classes separate game logic from RL training
- Wrapper-based reward shaping works with any game
- Gymnasium API compatibility maintained throughout

---

## Core Architecture Changes

### 1. Base Environment Class

**File**: `COOM/env/scenario.py`

#### Before (DoomEnv)
```python
import vizdoom as vzd
from vizdoom import ScreenResolution, GameVariable

class DoomEnv(BaseEnv):
    def __init__(self, ...):
        self.game = vzd.DoomGame()
        self.game.load_config(f"{scenario_dir}/conf.cfg")
        self.game.set_doom_scenario_path(f"{scenario_dir}/{env}.wad")
        self.game.init()

        # Game variables accessed by index
        vars_cur = game_var_buf[-1]
        var_cur = vars_cur[self.var_index]
```

#### After (MarioEnv)
```python
import retro

class MarioEnv(BaseEnv):
    def __init__(self, ...):
        # Set up custom integration
        integration_path = str(Path(__file__).parent.parent.parent / 'mario.stimuli')
        retro.data.Integrations.add_custom_path(integration_path)

        # Create retro environment
        self.game = retro.make(
            game='SuperMarioBros-Nes',
            state=env,  # e.g., 'Level1-1'
            inttype=retro.data.Integrations.CUSTOM,
            render_mode='rgb_array' if render else None
        )

        # Game variables accessed by name
        def _get_state_dict(self):
            state_dict = {}
            for var_name in ['score', 'coins', 'xscrollLo', ...]:
                state_dict[var_name] = float(self.game.data.lookup_value(var_name))
            return state_dict
```

#### Key Changes
1. **Initialization**: `.cfg` files → retro game/state specification
2. **Game State**: Indexed arrays → Named dictionary lookup
3. **Action Space**: Boolean arrays (same structure, different semantics)
4. **Observation**: 160×120 RGB → 224×256 RGB (NES resolution)

### 2. Game State Access

#### Before
```python
# ViZDoom: Access by index
health = state.game_variables[0]
x_pos = state.game_variables[1]
y_pos = state.game_variables[2]

# Store as list
self.game_variable_buffer.append(state.game_variables)
```

#### After
```python
# stable-retro: Access by name
health = self.game.data.lookup_value('lives')
x_pos = self.game.data.lookup_value('xscrollHi') * 256 + \
        self.game.data.lookup_value('xscrollLo')

# Store as dictionary
state_dict = self._get_state_dict()
self.game_variable_buffer.append(state_dict)
```

### 3. Reward System

**File**: `COOM/wrappers/reward.py`

#### Before
```python
from vizdoom import GameVariable

class GameVariableRewardWrapper(RewardWrapper):
    def __init__(self, env, reward: float, var_index: int = 0):
        self.var_index = var_index

    def reward(self, reward):
        var_cur = self.game_variable_buffer[-1][self.var_index]
        var_prev = self.game_variable_buffer[-2][self.var_index]
```

#### After
```python
class StateVariableRewardWrapper(RewardWrapper):
    def __init__(self, env, reward: float, var_name: str):
        self.var_name = var_name

    def reward(self, reward):
        var_cur = self.game_variable_buffer[-1].get(self.var_name, 0)
        var_prev = self.game_variable_buffer[-2].get(self.var_name, 0)
```

#### New Mario-Specific Wrappers
```python
class PositionRewardWrapper(RewardWrapper):
    """Reward for horizontal progress (x-position movement)"""
    def reward(self, reward):
        x_cur = vars_cur.get('xscrollHi', 0) * 256 + vars_cur.get('xscrollLo', 0)
        x_prev = vars_prev.get('xscrollHi', 0) * 256 + vars_prev.get('xscrollLo', 0)
        return reward + max(0, x_cur - x_prev) * self.scaler

class ScoreRewardWrapper(RewardWrapper):
    """Reward for increasing game score"""

class CoinRewardWrapper(RewardWrapper):
    """Reward for collecting coins"""

class TimeRewardWrapper(RewardWrapper):
    """Time penalty to encourage fast completion"""
```

---

## File-by-File Changes

### Modified Files

#### 1. `setup.py`
**Changes**:
- `vizdoom` → `stable-retro`
- Version: 1.0.1 → 2.0.0
- Description updated
- Keywords: "vizdoom" → "super mario bros", "stable-retro"

```diff
- coom_requirements = ["vizdoom", ...]
+ coom_requirements = ["stable-retro", ...]

- description="COOM: Benchmarking Continual Reinforcement Learning on Doom",
+ description="COOM: Benchmarking Continual Reinforcement Learning on Super Mario Bros",
```

#### 2. `COOM/env/scenario.py`
**Changes**: Complete rewrite
- `DoomEnv` → `MarioEnv`
- Removed ViZDoom imports
- Added retro integration setup
- Changed observation space: (160, 120, 3) → (224, 256, 3)
- Implemented `_get_state_dict()` for GameData conversion
- Updated `get_state_variable()` to use `lookup_value()`

**Lines changed**: ~350 lines (90% rewrite)

#### 3. `COOM/wrappers/reward.py`
**Changes**:
- `from gym import RewardWrapper` → `from gymnasium import RewardWrapper`
- Index-based access → Name-based access
- Removed `GameVariable` import
- Added new wrappers: `PositionRewardWrapper`, `ScoreRewardWrapper`, `CoinRewardWrapper`, `TimeRewardWrapper`, `DeathPenaltyWrapper`
- Updated all existing wrappers to use `var_name` instead of `var_index`

**Lines changed**: ~360 lines (80% rewrite)

#### 4. `COOM/wrappers/observation.py`
**Changes**:
- `from COOM.env.scenario import DoomEnv` → `from COOM.env.scenario import MarioEnv`
- Type hints updated

**Lines changed**: 2 lines

#### 5. `COOM/env/builder.py`
**Changes**:
- `DoomEnv` → `MarioEnv` in imports and type hints
- `doom_kwargs` → `mario_kwargs` throughout
- Action space builder updated for NES controller
- Added kwarg merging logic to preserve `action_space_fn`

**Action Space Before**:
```python
def build_multi_discrete_actions():
    # DOOM: Turn Left/Right, Move Forward, Execute (shoot/jump)
    # Returns 12 actions
    actions = []
    t_left_right = [[False, False], [False, True], [True, False]]
    m_forward = [[False], [True]]
    execute = [[False], [True]]
    # Combines into button arrays of length 4
```

**Action Space After**:
```python
def build_mario_actions():
    # NES Controller: [B, None, Select, Start, Up, Down, Left, Right, A]
    # Returns 12 actions: Movement (Left/Right/None) × Jump (A) × Run (B)
    # Combines into button arrays of length 9
    movement_options = [None, Left, Right]  # 3 options
    jump_options = [No, A]  # 2 options
    run_options = [No, B]  # 2 options
    # 3 × 2 × 2 = 12 actions
```

**Lines changed**: ~100 lines (50% rewrite)

#### 6. `COOM/env/continual.py`
**Changes**:
- `DoomEnv` → `MarioEnv` in imports and type hints
- `doom_config` → `mario_config` parameter
- Documentation updated

**Lines changed**: 5 lines

#### 7. `COOM/utils/config.py`
**Changes**: Complete rewrite
- Removed all DOOM scenario imports
- Added 8 Mario world imports (World1-8)
- Removed DOOM sequences (CD4, CD8, CO4, CO8, etc.)
- Added Mario sequences:
  - `WORLD_PROGRESSION_4`
  - `WORLD_PROGRESSION_8`
  - `STAGE_TYPES_4`
  - `WORLD_COMPLETE`
  - `DIFFICULTY_CURVE`
  - `MIXED_WORLDS`
- Updated `scenario_config` dictionary
- Added level name mappings

**Before**:
```python
class Scenario(Enum):
    PITFALL = 1
    ARMS_DEALER = 2
    FLOOR_IS_LAVA = 3
    # ... 8 DOOM scenarios

class Sequence(Enum):
    CD4 = 1   # Cross-Domain 4 tasks
    CD8 = 2   # Cross-Domain 8 tasks
    CO4 = 3   # Cross-Objective 4 tasks
    # ...
```

**After**:
```python
class Scenario(Enum):
    WORLD1 = 1
    WORLD2 = 2
    # ... 8 Mario worlds

class Sequence(Enum):
    WORLD_PROGRESSION_4 = 1
    WORLD_PROGRESSION_8 = 2
    STAGE_TYPES_4 = 3
    # ... Mario-specific sequences
```

**Lines changed**: ~180 lines (95% rewrite)

#### 8. `COOM/utils/utils.py`
**Changes**:
- Removed ViZDoom imports (`ScreenResolution`)
- Removed `get_screen_resolution()` function
- Updated `distance_traversed()` to use named variables
- Added Mario-specific helpers:
  - `get_x_position(state)` - Combines Hi/Lo bytes
  - `get_player_position(state)` - Returns (x, y) tuple

**Before**:
```python
from vizdoom import ScreenResolution

def get_screen_resolution(resolution: str) -> ScreenResolution:
    return resolutions[resolution]

def distance_traversed(game_var_buf, x_index: int, y_index: int):
    x_cur = game_var_buf[-1][x_index]
    y_cur = game_var_buf[-1][y_index]
```

**After**:
```python
def distance_traversed(game_var_buf, x_var: str, y_var: str):
    x_cur = game_var_buf[-1].get(x_var, 0)
    y_cur = game_var_buf[-1].get(y_var, 0)

def get_x_position(state: Dict) -> int:
    return int(state.get('xscrollHi', 0) * 256 + state.get('xscrollLo', 0))
```

**Lines changed**: ~75 lines (100% rewrite)

#### 9. `COOM/examples/run_single.py`
**Changes**:
- Updated scenario choices: DOOM scenarios → Mario worlds
- Updated task parameter: `'default'/'hard'` → `'Level1-1'` etc.
- Added `--max-steps` parameter
- Updated help text

**Lines changed**: ~15 lines

#### 10. `COOM/examples/run_sequence.py`
**Changes**:
- Updated sequence choices: CD/CO sequences → Mario sequences
- Added better output formatting
- Updated help text

**Lines changed**: ~20 lines

### New Files

#### 1. `COOM/env/scenarios/world1/world1.py` (×8 for all worlds)
**Purpose**: Scenario classes for each Mario world

**Structure**:
```python
class World1(MarioEnv):
    TASKS = ['Level1-1', 'Level1-2', 'Level1-3', 'Level1-4']

    def __init__(self, mario_kwargs, ...):
        super().__init__(**mario_kwargs)
        # Statistics tracking
        self.max_x_position = 0
        self.total_coins = 0

    def store_statistics(self, game_var_buf):
        # Track max X position, coins, score

    def get_success_metric(self) -> float:
        return float(self.max_x_position)

    def reward_wrappers_dense(self):
        return [
            WrapperHolder(PositionRewardWrapper, scaler=1.0),
            WrapperHolder(ConstantRewardWrapper, reward=-0.01)
        ]

    @property
    def performance_upper_bound(self) -> float:
        # Level-specific bounds for normalization
        return 3200.0
```

**Total**: 8 files, ~120 lines each = ~960 lines

#### 2. `COOM/env/scenarios/world1/__init__.py` (×8)
```python
from .world1 import World1
```

#### 3. `test_setup.py`
**Purpose**: Diagnostic script to verify installation
- Tests all imports
- Checks stable-retro installation
- Verifies mario.stimuli integration
- Provides helpful error messages

**Lines**: ~150 lines

#### 4. `MIGRATION.md`
**Purpose**: This file!

### Deleted/Obsolete Files

The following DOOM scenario files are no longer used (but kept for reference):
- `COOM/env/scenarios/health_gathering/`
- `COOM/env/scenarios/run_and_gun/`
- `COOM/env/scenarios/chainsaw/`
- `COOM/env/scenarios/raise_the_roof/`
- `COOM/env/scenarios/floor_is_lava/`
- `COOM/env/scenarios/hide_and_seek/`
- `COOM/env/scenarios/arms_dealer/`
- `COOM/env/scenarios/pitfall/`
- `COOM/env/scenarios/parkour/`

These still contain ViZDoom imports and won't work, but are preserved as reference.

---

## API Changes

### Environment Creation

#### Before
```python
from COOM.env.builder import make_env
from COOM.utils.config import Scenario

env = make_env(
    Scenario.HEALTH_GATHERING,
    task='hard',
    doom_kwargs={'render': True}
)
```

#### After
```python
from COOM.env.builder import make_env
from COOM.utils.config import Scenario

env = make_env(
    Scenario.WORLD1,
    task='Level1-1',
    mario_kwargs={'render': True}
)
```

### Continual Learning Sequences

#### Before
```python
from COOM.utils.config import Sequence

cl_env = ContinualLearningEnv(
    Sequence.CO8,  # Cross-Objective 8 tasks
    steps_per_env=2e5
)
# 8 different DOOM scenarios, 1 task each
```

#### After
```python
from COOM.utils.config import Sequence

cl_env = ContinualLearningEnv(
    Sequence.WORLD_PROGRESSION_8,
    steps_per_env=2e5
)
# 8 Mario worlds (Level1-1 through Level8-1)
```

### Custom Reward Wrappers

#### Before
```python
from COOM.wrappers.reward import GameVariableRewardWrapper

# Reward for health increase (variable index 0)
wrapper = GameVariableRewardWrapper(env, reward=15.0, var_index=0)
```

#### After
```python
from COOM.wrappers.reward import StateVariableRewardWrapper

# Reward for coin collection
wrapper = StateVariableRewardWrapper(env, reward=10.0, var_name='coins')

# Or use Mario-specific wrappers
from COOM.wrappers.reward import PositionRewardWrapper, CoinRewardWrapper

position_wrapper = PositionRewardWrapper(env, scaler=1.0)
coin_wrapper = CoinRewardWrapper(env, reward=10.0)
```

### Statistics Access

#### Before
```python
# DOOM: Different metrics per scenario
stats = env.extra_statistics('train')
# health_gathering: {'train/kits_obtained', 'train/movement'}
# run_and_gun: {'train/kills', 'train/ammo_used'}
```

#### After
```python
# Mario: Consistent metrics across all worlds
stats = env.extra_statistics('train')
# All worlds: {
#   'train/max_x_position',
#   'train/coins',
#   'train/score',
#   'train/frames'
# }
```

---

## Configuration Changes

### Scenario Configuration

#### Before (`DOOM/env/scenarios/*/conf.cfg`)
```ini
screen_resolution = RES_160X120
screen_format = CRCGCB
episode_timeout = 2500
available_buttons = { TURN_LEFT TURN_RIGHT MOVE_FORWARD SPEED }
available_game_variables = { HEALTH POSITION_X POSITION_Y }
```

#### After (`mario.stimuli/SuperMarioBros-Nes/`)

**data.json**: Memory addresses for game variables
```json
{
  "info": {
    "score": {"address": 2013, "type": ">n6"},
    "coins": {"address": 1886, "type": "|u1"},
    "xscrollLo": {"address": 1820, "type": "|u1"},
    "xscrollHi": {"address": 1818, "type": "|u1"},
    "lives": {"address": 1882, "type": "|i1"}
  }
}
```

**scenario.json**: Reward and done conditions
```json
{
  "done": {
    "variables": {
      "lives": {"op": "equal", "reference": -1},
      "stage": {"measurement": "delta", "op": "positive"}
    }
  },
  "reward": {
    "variables": {
      "xscrollLo": {"reward": 1}
    }
  }
}
```

### Wrapper Configuration

**No changes** - The wrapper configuration remains the same:
```python
default_wrapper_config = {
    'augment': False,
    'resize': True,
    'frame_height': 84,
    'frame_width': 84,
    'rescale': True,
    'normalize_observation': True,
    'frame_stack': 4,
    'lstm': False,
    'record': False,
    'sparse_rewards': False,
}
```

---

## Dependencies

### Before (`setup.py`)
```python
coom_requirements = [
    "vizdoom",           # DOOM game engine
    "opencv-python",     # Image processing
    "scipy==1.11.4",     # Utilities
    "gymnasium==0.28.1"  # RL API
]
```

### After (`setup.py`)
```python
coom_requirements = [
    "stable-retro",      # NES/Retro game emulator
    "opencv-python",     # Image processing (unchanged)
    "scipy==1.11.4",     # Utilities (unchanged)
    "gymnasium==0.28.1"  # RL API (unchanged)
]
```

### Installation Changes

#### Before
```bash
pip install COOM  # Installs vizdoom automatically
# DOOM WAD files included in package
```

#### After
```bash
pip install -e .  # Installs stable-retro
python -m retro.import /path/to/roms/  # Must import SMB ROM
# mario.stimuli integration included in repo
```

---

## Key Differences

### 1. Game Mechanics

| Aspect | DOOM | Super Mario Bros |
|--------|------|------------------|
| **Genre** | First-person shooter | 2D platformer |
| **Perspective** | Egocentric 3D | Side-scrolling 2D |
| **Primary Action** | Shooting | Jumping |
| **Success Metric** | Scenario-dependent (kills, survival, items) | Horizontal progress (x-position) |
| **Episode Length** | 1000-2500 steps | Variable (until death/goal) |

### 2. Observation Space

| Property | DOOM | Mario |
|----------|------|-------|
| **Resolution** | 160×120 pixels | 224×256 pixels (NES native) |
| **Aspect Ratio** | 4:3 | 7:8 (slightly vertical) |
| **Channels** | RGB (3) | RGB (3) |
| **Size** | 57,600 values | 172,032 values (~3× larger) |

### 3. Action Space

Both use 12 discrete actions, but with different semantics:

| DOOM | Mario |
|------|-------|
| No turn + No move + No action | No move + No jump + No run |
| No turn + No move + Action (shoot) | No move + No jump + Run |
| No turn + Move forward + No action | No move + Jump + No run |
| No turn + Move forward + Action | No move + Jump + Run |
| Turn left + No move + No action | Left + No jump + No run |
| Turn left + No move + Action | Left + No jump + Run |
| Turn left + Move forward + No action | Left + Jump + No run |
| Turn left + Move forward + Action | Left + Jump + Run |
| Turn right + No move + No action | Right + No jump + No run |
| Turn right + No move + Action | Right + No jump + Run |
| Turn right + Move forward + No action | Right + Jump + No run |
| Turn right + Move forward + Action | Right + Jump + Run |

### 4. State Variables

#### DOOM
- **Types**: HEALTH, KILLCOUNT, AMMO, POSITION_X, POSITION_Y
- **Access**: By GameVariable enum
- **Count**: 3-5 per scenario
- **Format**: Float array

#### Mario
- **Types**: score, coins, lives, xscroll, player_pos, powerstate, time, etc.
- **Access**: By string name
- **Count**: 36 tracked variables
- **Format**: Dictionary with various types

### 5. Reward Structure

#### DOOM (Diverse)
Each scenario has unique rewards:
- **Health Gathering**: +15 per health kit, -0.01 per step
- **Run and Gun**: +1 per kill
- **Pitfall**: +1 per platform reached
- **Chainsaw**: +10 per kill

#### Mario (Unified)
All worlds use same reward structure:
- **Dense**: +1.0 × x_position_delta - 0.01 per step
- **Sparse**: +1.0 × x_position_delta

### 6. Task Variety

#### DOOM
- **Between scenarios**: Different objectives (kill, survive, navigate, collect)
- **Within scenario**: Visual/dynamic variations (textures, enemy types, layouts)

#### Mario
- **Between worlds**: Different visual themes and enemy types
- **Within world**: Progressive difficulty (stages 1-4)
- **Consistent objective**: Reach the end of the level

---

## Migration Checklist

### For Users Migrating Code

- [ ] Replace `Scenario.HEALTH_GATHERING` → `Scenario.WORLD1` (etc.)
- [ ] Replace task names: `'default'/'hard'` → `'Level1-1'` (etc.)
- [ ] Replace sequence types: `Sequence.CO8` → `Sequence.WORLD_PROGRESSION_8`
- [ ] Update reward wrapper imports:
  - `GameVariableRewardWrapper` → `StateVariableRewardWrapper`
  - Or use new: `PositionRewardWrapper`, `CoinRewardWrapper`
- [ ] Update reward wrapper parameters: `var_index=0` → `var_name='coins'`
- [ ] Update `doom_kwargs` → `mario_kwargs` in environment creation
- [ ] Install `stable-retro` instead of `vizdoom`
- [ ] Import Super Mario Bros ROM file
- [ ] Update observation size expectations: (160, 120, 3) → (224, 256, 3)

### For Developers Extending COOM

If creating new scenarios:

1. **Create scenario directory**: `COOM/env/scenarios/worldX/`
2. **Create scenario class**: Inherit from `MarioEnv`
3. **Define TASKS**: List of level names (e.g., `['LevelX-1', 'LevelX-2', ...]`)
4. **Implement methods**:
   - `store_statistics()` - Track episode statistics
   - `get_success_metric()` - Return primary success measure
   - `reward_wrappers_dense()` - Define dense reward structure
   - `reward_wrappers_sparse()` - Define sparse reward structure
   - `extra_statistics()` - Return additional metrics
   - `performance_upper_bound` - Max expected performance
   - `performance_lower_bound` - Baseline (random) performance

5. **Create level state files** in `mario.stimuli/`:
   - Use `generate_sublevels.py` as template
   - Create `.state` files for each level

6. **Update `config.py`**:
   - Add scenario to `Scenario` enum
   - Add to `scenario_config` dict
   - Update sequences as needed

---

## Summary Statistics

### Code Changes
- **Files modified**: 10 core files
- **Files created**: 17 new files (8 worlds + 8 __init__ + test_setup.py)
- **Lines changed**: ~2,500 lines
- **Percentage rewritten**: ~60% of core codebase

### Preserved Architecture
- ✅ BaseEnv abstract class
- ✅ Gymnasium API compatibility
- ✅ Wrapper-based reward shaping
- ✅ Observation preprocessing pipeline
- ✅ Continual learning sequences
- ✅ Statistics tracking framework
- ✅ Builder pattern for environment creation
- ✅ SAC training infrastructure (in `CL/` directory)

### New Capabilities
- ✅ NES game emulation via stable-retro
- ✅ Custom retro integration support
- ✅ 32 unique Mario levels (vs. ~20 DOOM tasks)
- ✅ Unified reward structure across scenarios
- ✅ High-resolution observations (224×256 vs. 160×120)
- ✅ Platform game mechanics
- ✅ Mario-specific continual learning sequences

---

## Troubleshooting

### Common Issues After Migration

1. **Import Error: `No module named 'gym'`**
   - **Fix**: Changed to `gymnasium` in `COOM/wrappers/reward.py`

2. **AttributeError: `'GameData' object has no attribute 'copy'`**
   - **Fix**: Use `_get_state_dict()` method instead of `.copy()`

3. **TypeError: `'NoneType' object is not callable`**
   - **Fix**: Ensure `action_space_fn` is in `mario_kwargs`
   - Default value now properly merged in `make_env()`

4. **FileNotFoundError: ROM not found**
   - **Fix**: Import ROM with `python -m retro.import /path/to/roms/`

5. **ImportError: Cannot import name 'DoomEnv'**
   - **Fix**: All references changed to `MarioEnv`

---

## Future Extensions

### Easy Additions
- [ ] More continual learning sequences (difficulty-based, stage-type-based)
- [ ] Additional reward wrappers (enemy kills, powerup collection)
- [ ] Support for other retro games (Sonic, Mega Man, etc.)

### Medium Complexity
- [ ] Multi-task learning across different game types
- [ ] Transfer learning from Mario to other platformers
- [ ] Curriculum learning based on level difficulty

### Advanced
- [ ] Procedurally generated Mario levels
- [ ] Meta-learning across game mechanics
- [ ] Cross-game continual learning benchmarks

---

## Conclusion

The migration from DOOM to Super Mario Bros demonstrates COOM's modular architecture. By isolating game-specific code in scenario classes and using abstract interfaces, the core RL infrastructure remains unchanged while the underlying game can be completely swapped.

**Key Takeaway**: The same continual learning algorithms, training procedures, and evaluation metrics work seamlessly with both DOOM and Mario, validating the benchmark's design principles.

For questions or issues, please refer to:
- `README.md` - Installation and usage
- `test_setup.py` - Diagnostic tool
- GitHub Issues - Bug reports and feature requests
