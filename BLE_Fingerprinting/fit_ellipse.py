import math
from dataclasses import dataclass

import numpy as np


@dataclass
class EllipseFit:
    a: float
    b: float
    phi: float
    X0: float
    Y0: float
    X0_in: float
    Y0_in: float
    long_axis: float
    short_axis: float
    status: str


def fit_ellipse(x: np.ndarray, y: np.ndarray) -> EllipseFit:
    """Port of the MATLAB fit_ellipse utility from MathWorks File Exchange."""
    orientation_tolerance = 1e-7

    x = np.asarray(x).reshape(-1)
    y = np.asarray(y).reshape(-1)

    mean_x = np.mean(x)
    mean_y = np.mean(y)

    X = np.column_stack((x**2, x * y, y**2, x, y))
    a_vec = np.sum(X, axis=0) @ np.linalg.inv(X.T @ X)

    a, b, c, d, e = a_vec

    if min(abs(b / a), abs(b / c)) > orientation_tolerance:
        orientation_rad = 0.5 * math.atan2(b, (c - a))
        cos_phi = math.cos(orientation_rad)
        sin_phi = math.sin(orientation_rad)
        a, b, c, d, e = (
            a * cos_phi**2 - b * cos_phi * sin_phi + c * sin_phi**2,
            0.0,
            a * sin_phi**2 + b * cos_phi * sin_phi + c * cos_phi**2,
            d * cos_phi - e * sin_phi,
            d * sin_phi + e * cos_phi,
        )
        mean_x, mean_y = (
            cos_phi * mean_x - sin_phi * mean_y,
            sin_phi * mean_x + cos_phi * mean_y,
        )
    else:
        orientation_rad = 0.0
        cos_phi = math.cos(orientation_rad)
        sin_phi = math.sin(orientation_rad)

    test = a * c
    if test <= 0:
        status = "Parabola found" if test == 0 else "Hyperbola found"
        return EllipseFit(
            a=np.nan,
            b=np.nan,
            phi=np.nan,
            X0=np.nan,
            Y0=np.nan,
            X0_in=np.nan,
            Y0_in=np.nan,
            long_axis=np.nan,
            short_axis=np.nan,
            status=status,
        )

    if a < 0:
        a, c, d, e = -a, -c, -d, -e

    X0 = -d / (2 * a)
    Y0 = -e / (2 * c)
    F = 1 + (d**2) / (4 * a) + (e**2) / (4 * c)
    a_axis = math.sqrt(abs(F / a))
    b_axis = math.sqrt(abs(F / c))
    long_axis = 2 * max(a_axis, b_axis)
    short_axis = 2 * min(a_axis, b_axis)

    R = np.array([[cos_phi, sin_phi], [-sin_phi, cos_phi]])
    X0_in, Y0_in = (R @ np.array([X0, Y0])).tolist()

    return EllipseFit(
        a=a_axis,
        b=b_axis,
        phi=orientation_rad,
        X0=X0,
        Y0=Y0,
        X0_in=X0_in,
        Y0_in=Y0_in,
        long_axis=long_axis,
        short_axis=short_axis,
        status="",
    )
