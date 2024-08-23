import numpy as np


def distance_point_to_segment(P, A, B):
    """
    Compute the distance between points and a segment in 2D.

    Parameters:
        P (np.ndarray): An array of shape (n, m) where n is the number of points and m is the number of dimensions (2 or more).
        A (np.ndarray): A 1D array representing one endpoint of the segment.
        B (np.ndarray): A 1D array representing the other endpoint of the segment.

    Returns:
        np.ndarray: An array of shape (n,) where each element is the distance from the corresponding point in P to the segment AB.
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
