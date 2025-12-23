"""
Module for geometric calculations in 2D and 3D spaces.

This module provides utility functions for geometric calculations.
It include the computation of distances between points and line segments in 2D and 3D spaces.
It is used to remove the crack from the pacman region in SIF estimation.

Functions:
    distance_point_to_segment(P, A, B): Computes the distance between points and a line segment.
"""

import numpy as np


def distance_point_to_segment(
    P: np.ndarray, A: np.ndarray, B: np.ndarray
) -> np.ndarray:
    """Computes the distance between points and a line segment in 2D or 3D space.

    This function calculates the shortest distance from each point in array P to the line segment defined by endpoints A and B.

    Args:
        P (np.ndarray):
            Array of shape (n, m) where n is the number of points and m is the number of dimensions (2 or 3).
            Each row represents a point.
        A (np.ndarray):
            1D array of shape (m,) representing one endpoint of the segment.
        B (np.ndarray):
            1D array of shape (m,) representing the other endpoint of the segment.

    Returns:
        distance (np.ndarray):
            Array of shape (n,) where each element is the distance from the corresponding point in P to the segment AB.
    """
    # Ensure A and B are 2D for broadcasting with P
    A = A[:, np.newaxis] if A.ndim == 1 else A
    B = B[:, np.newaxis] if B.ndim == 1 else B

    # Vector AB
    AB = B - A

    # Vector AP for each point P
    AP = P - A

    # Project vector AP onto AB to find the projection point on the line (may lie outside the segment)
    AB_squared = np.sum(AB**2, axis=0)
    if np.all(AB_squared == 0):  # A and B are the same point
        return np.linalg.norm(AP, axis=0)

    t = np.clip(np.sum(AP * AB, axis=0) / AB_squared, 0, 1)

    # The projection point on the segment
    projection = A + t * AB

    # Distance from point P to the projection point
    distance = np.linalg.norm(P - projection, axis=0)

    return distance
