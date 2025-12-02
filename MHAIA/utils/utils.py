from collections import deque
from typing import Dict

import numpy as np
from scipy import spatial


def distance_traversed(game_var_buf: deque, x_var: str, y_var: str) -> float:
    """
    Calculate Euclidean distance traveled between first and last state in buffer.

    Args:
        game_var_buf: Deque of game state dictionaries
        x_var: Name of x-position variable (e.g., 'xscrollLo')
        y_var: Name of y-position variable (e.g., 'player_y_pos')

    Returns:
        Euclidean distance between first and last positions
    """
    if len(game_var_buf) < 2:
        return 0.0

    coordinates_curr = [
        game_var_buf[-1].get(x_var, 0),
        game_var_buf[-1].get(y_var, 0)
    ]
    coordinates_past = [
        game_var_buf[0].get(x_var, 0),
        game_var_buf[0].get(y_var, 0)
    ]
    return spatial.distance.euclidean(coordinates_curr, coordinates_past)


def get_x_position(state: Dict) -> int:
    """
    Get full x-position from Mario game state by combining Hi and Lo bytes.

    Args:
        state: Game state dictionary

    Returns:
        Full x-position as integer
    """
    x_hi = state.get('xscrollHi', 0)
    x_lo = state.get('xscrollLo', 0)
    return int(x_hi * 256 + x_lo)


def get_player_position(state: Dict) -> tuple:
    """
    Get player (x, y) position from Mario game state.

    Args:
        state: Game state dictionary

    Returns:
        Tuple of (x, y) positions
    """
    x_hi = state.get('player_x_posHi', 0)
    x_lo = state.get('player_x_posLo', 0)
    x = int(x_hi * 256 + x_lo)
    y = int(state.get('player_y_pos', 0))
    return (x, y)


def combine_frames(obs):
    """
    Combine multiple frame observations along the channel dimension.

    Args:
        obs: List or array of observation frames

    Returns:
        Concatenated observation
    """
    return np.concatenate(obs, axis=2)
