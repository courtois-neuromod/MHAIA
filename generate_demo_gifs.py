#!/usr/bin/env python3
"""Generate demonstration GIFs for GHAIA README."""

import argparse
import os
from pathlib import Path

import gymnasium as gym
import numpy as np
from PIL import Image

from GHAIA.env.builder import make_env
from GHAIA.utils.config import Scenario


def generate_gif(scenario: Scenario, task: str, output_path: str, max_steps: int = 500, fps: int = 15):
    """Generate a GIF from Mario gameplay.

    Args:
        scenario: The world scenario to run
        task: The level name (e.g., 'Level1-1')
        output_path: Path to save the GIF
        max_steps: Maximum number of steps to record
        fps: Frames per second for the GIF
    """
    # Create environment with rendering enabled
    env = make_env(scenario, task, mario_kwargs={'render': True})

    frames = []
    obs, info = env.reset()

    print(f"Recording {task} gameplay...")
    step = 0
    while step < max_steps:
        # Use random actions for demo (you could load a trained agent here)
        action = env.action_space.sample()
        obs, reward, done, truncated, info = env.step(action)

        # Capture frame every few steps to keep GIF size manageable
        if step % 2 == 0:  # Capture every other frame
            frame = env.render()
            if frame is not None:
                # Handle case where render returns a list
                if isinstance(frame, list) and len(frame) > 0:
                    frame = frame[0]
                if not isinstance(frame, list):
                    frames.append(frame)

        step += 1

        if done or truncated:
            print(f"Episode ended after {step} steps")
            break

    env.close()

    # Save frames as GIF
    if frames:
        print(f"Saving GIF with {len(frames)} frames to {output_path}...")
        images = [Image.fromarray(frame) for frame in frames]

        # Calculate duration per frame in milliseconds
        duration_ms = int(1000 / fps)

        # Save as GIF
        images[0].save(
            output_path,
            save_all=True,
            append_images=images[1:],
            duration=duration_ms,
            loop=0,
            optimize=True
        )
        print(f"GIF saved successfully!")
    else:
        print("No frames captured!")


def main(args):
    # Create output directory if it doesn't exist
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate demo GIFs for different levels
    demos = [
        (Scenario.WORLD1, 'Level1-1', 'mario_demo1.gif'),
        (Scenario.WORLD3, 'Level3-1', 'mario_demo2.gif'),
    ]

    for scenario, task, filename in demos:
        output_path = output_dir / filename
        generate_gif(scenario, task, str(output_path), args.max_steps, args.fps)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate demo GIFs for GHAIA")
    parser.add_argument('--output-dir', type=str, default='assets/gifs',
                        help="Directory to save GIFs")
    parser.add_argument('--max-steps', type=int, default=500,
                        help="Maximum steps per episode")
    parser.add_argument('--fps', type=int, default=15,
                        help="Frames per second for GIF")

    args = parser.parse_args()
    main(args)
