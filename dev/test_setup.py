#!/usr/bin/env python3
"""
Simple diagnostic script to test Mario COOM setup.
Run this to verify all dependencies are correctly installed.
"""

import sys

print("=" * 60)
print("MHAIA Setup Diagnostic")
print("=" * 60)
print()

# Check Python version
print(f"Python version: {sys.version}")
print()

# Test imports
print("Testing imports...")
errors = []

try:
    import numpy as np
    print("✓ numpy")
except ImportError as e:
    print("✗ numpy")
    errors.append(("numpy", str(e)))

try:
    import gymnasium
    print("✓ gymnasium")
except ImportError as e:
    print("✗ gymnasium")
    errors.append(("gymnasium", str(e)))

try:
    import retro
    print("✓ stable-retro")
    print(f"  retro version: {retro.__version__ if hasattr(retro, '__version__') else 'unknown'}")
except ImportError as e:
    print("✗ stable-retro (REQUIRED!)")
    errors.append(("stable-retro", str(e)))

try:
    import cv2
    print("✓ opencv-python")
except ImportError as e:
    print("✗ opencv-python")
    errors.append(("opencv-python", str(e)))

try:
    import scipy
    print("✓ scipy")
except ImportError as e:
    print("✗ scipy")
    errors.append(("scipy", str(e)))

print()

# Test MHAIA imports
print("Testing MHAIA modules...")
try:
    from MHAIA.env.scenario import MarioEnv
    print("✓ MHAIA.env.scenario.MarioEnv")
except ImportError as e:
    print("✗ MHAIA.env.scenario.MarioEnv")
    errors.append(("MHAIA.env.scenario", str(e)))

try:
    from MHAIA.env.builder import make_env, build_mario_actions
    print("✓ MHAIA.env.builder")
except ImportError as e:
    print("✗ MHAIA.env.builder")
    errors.append(("MHAIA.env.builder", str(e)))

try:
    from MHAIA.utils.config import Scenario, Sequence
    print("✓ MHAIA.utils.config")
except ImportError as e:
    print("✗ MHAIA.utils.config")
    errors.append(("MHAIA.utils.config", str(e)))

try:
    from MHAIA.wrappers.reward import PositionRewardWrapper
    print("✓ MHAIA.wrappers.reward")
except ImportError as e:
    print("✗ MHAIA.wrappers.reward")
    errors.append(("MHAIA.wrappers.reward", str(e)))

print()

# Check retro integration
if 'retro' in sys.modules:
    print("Checking stable-retro integration...")
    import retro
    from pathlib import Path

    # Check if custom integration path exists
    mario_stimuli_path = Path(__file__).parent / 'mario.stimuli'
    if mario_stimuli_path.exists():
        print(f"✓ mario.stimuli found at: {mario_stimuli_path}")

        # Check for key files
        smb_path = mario_stimuli_path / 'SuperMarioBros-Nes'
        if smb_path.exists():
            print("✓ SuperMarioBros-Nes integration found")

            data_json = smb_path / 'data.json'
            scenario_json = smb_path / 'scenario.json'

            if data_json.exists():
                print("  ✓ data.json")
            else:
                print("  ✗ data.json missing")
                errors.append(("mario.stimuli", "data.json not found"))

            if scenario_json.exists():
                print("  ✓ scenario.json")
            else:
                print("  ✗ scenario.json missing")
                errors.append(("mario.stimuli", "scenario.json not found"))
        else:
            print("✗ SuperMarioBros-Nes integration not found")
            errors.append(("mario.stimuli", "SuperMarioBros-Nes directory not found"))
    else:
        print("✗ mario.stimuli directory not found")
        errors.append(("mario.stimuli", "Directory not found"))

    print()

# Summary
print("=" * 60)
if errors:
    print("SETUP INCOMPLETE - Please fix the following issues:")
    print()
    for module, error in errors:
        print(f"  • {module}: {error}")
    print()
    print("Installation instructions:")
    print("  pip install stable-retro opencv-python scipy gymnasium")
    sys.exit(1)
else:
    print("✓ ALL CHECKS PASSED!")
    print()
    print("Your setup is ready. Try running:")
    print("  python MHAIA/examples/run_single.py --scenario world1 --task Level1-1")
    sys.exit(0)
