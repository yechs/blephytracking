"""
Fit an ellipse to a 2-D point cloud using least-squares conic-section fitting.

Port of the MATLAB fit_ellipse by Ohad Gal (MathWorks File Exchange #3215).
Returns a dataclass with the same fields as the original MATLAB struct.
Raises ValueError if the conic is not an ellipse (parabola / hyperbola detected).
"""
from dataclasses import dataclass
from typing import Optional
import numpy as np


@dataclass
class Ellipse:
    a: float          # semi-axis in X direction (non-tilted frame)
    b: float          # semi-axis in Y direction (non-tilted frame)
    phi: float        # orientation angle (radians)
    X0: float         # centre X (non-tilted frame)
    Y0: float         # centre Y (non-tilted frame)
    X0_in: float      # centre X (original / tilted frame)
    Y0_in: float      # centre Y (original / tilted frame)
    long_axis: float
    short_axis: float
    status: str = ""


def fit_ellipse(x, y) -> Ellipse:
    """
    Fit an ellipse to points (x, y).  At least 5 points are required.

    Raises
    ------
    ValueError
        If the fitted conic is a parabola or hyperbola rather than an ellipse.
    """
    ORIENTATION_TOL = 1e-7

    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()

    # Build design matrix  [x² , x·y , y² , x , y]
    X = np.column_stack([x**2, x * y, y**2, x, y])

    # Least-squares: a = sum(X) / (X'*X)
    # sum(X) is a row vector (1×5); X'*X is (5×5)
    # Solve:  (X'*X) * a'  =  sum(X)'
    try:
        coeffs = np.linalg.solve(X.T @ X, X.sum(axis=0))
    except np.linalg.LinAlgError:
        raise ValueError("Matrix inversion failed during ellipse fitting")

    a, b, c, d, e = coeffs

    # Remove orientation (tilt) if B coefficient is significant
    if min(abs(b / a) if a != 0 else np.inf,
           abs(b / c) if c != 0 else np.inf) > ORIENTATION_TOL:
        orientation_rad = 0.5 * np.arctan(b / (c - a))
        cos_p = np.cos(orientation_rad)
        sin_p = np.sin(orientation_rad)
        a, b, c, d, e = (
            a * cos_p**2 - b * cos_p * sin_p + c * sin_p**2,
            0.0,
            a * sin_p**2 + b * cos_p * sin_p + c * cos_p**2,
            d * cos_p - e * sin_p,
            d * sin_p + e * cos_p,
        )
        mean_x = cos_p * np.mean(x) - sin_p * np.mean(y)
        mean_y = sin_p * np.mean(x) + cos_p * np.mean(y)
    else:
        orientation_rad = 0.0
        cos_p = 1.0
        sin_p = 0.0
        mean_x = np.mean(x)
        mean_y = np.mean(y)

    # Conic classification
    test = a * c
    if test == 0:
        raise ValueError("Parabola found – not an ellipse")
    if test < 0:
        raise ValueError("Hyperbola found – not an ellipse")

    # Ensure positive coefficients
    if a < 0:
        a, c, d, e = -a, -c, -d, -e

    # Centre of the non-tilted ellipse
    X0 = -d / (2.0 * a)
    Y0 = -e / (2.0 * c)

    # F'' for axis lengths
    F_pp = 1.0 + (d**2) / (4.0 * a) + (e**2) / (4.0 * c)
    a_ax = np.sqrt(abs(F_pp / a))
    b_ax = np.sqrt(abs(F_pp / c))

    long_axis = 2.0 * max(a_ax, b_ax)
    short_axis = 2.0 * min(a_ax, b_ax)

    # Rotate centre back to the original (tilted) frame
    R = np.array([[cos_p, sin_p], [-sin_p, cos_p]])
    P_in = R @ np.array([X0, Y0])
    X0_in, Y0_in = P_in

    return Ellipse(
        a=a_ax,
        b=b_ax,
        phi=orientation_rad,
        X0=X0,
        Y0=Y0,
        X0_in=X0_in,
        Y0_in=Y0_in,
        long_axis=long_axis,
        short_axis=short_axis,
        status="",
    )
