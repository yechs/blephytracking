import numpy as np
import warnings


def fit_ellipse(x, y, orientation_tolerance=1e-7):
    x = np.asarray(x).ravel()
    y = np.asarray(y).ravel()
    if x.size < 5:
        raise ValueError("Need at least 5 points")

    # X @ p ≈ 1, where p = [A,B,C,D,E] and A x^2 + B x y + C y^2 + D x + E y ≈ 1
    X = np.column_stack([x**2, x * y, y**2, x, y])
    ones = np.ones_like(x)

    # More stable than solving (X.T@X)p = X.T@1
    p, *_ = np.linalg.lstsq(X, ones, rcond=None)
    A, B, C, D, E = p

    # Remove orientation if needed
    if (
        min(abs(B / A) if A != 0 else np.inf, abs(B / C) if C != 0 else np.inf)
        > orientation_tolerance
    ):
        phi = 0.5 * np.arctan2(B, (C - A))
        cphi, sphi = np.cos(phi), np.sin(phi)

        At = A * cphi**2 - B * cphi * sphi + C * sphi**2
        Bt = 0.0
        Ct = A * sphi**2 + B * cphi * sphi + C * cphi**2
        Dt = D * cphi - E * sphi
        Et = D * sphi + E * cphi

        A, B, C, D, E = At, Bt, Ct, Dt, Et
    else:
        phi = 0.0
        cphi, sphi = 1.0, 0.0

    # Check ellipse (in this rotated form, need A and C same sign)
    test = A * C
    if test <= 0:
        status = "Parabola found" if test == 0 else "Hyperbola found"
        warnings.warn(f"fit_ellipse: Did not locate an ellipse ({status})")
        return {
            "a": None,
            "b": None,
            "phi": None,
            "X0": None,
            "Y0": None,
            "X0_in": None,
            "Y0_in": None,
            "long_axis": None,
            "short_axis": None,
            "status": status,
        }

    # Make coefficients positive as MATLAB does
    if A < 0:
        A, C, D, E = -A, -C, -D, -E

    # Center in the non-tilted (rotated) coordinates
    X0 = -D / (2 * A)
    Y0 = -E / (2 * C)

    # Same F expression as MATLAB (because we're fitting ... = 1)
    F = 1.0 + (D**2) / (4 * A) + (E**2) / (4 * C)

    # Require real axes
    if F <= 0 or (F / A) <= 0 or (F / C) <= 0:
        warnings.warn(
            "fit_ellipse: non-real ellipse parameters (check data / outliers)"
        )
        return None

    a_axis = np.sqrt(F / A)
    b_axis = np.sqrt(F / C)

    long_axis = 2 * max(a_axis, b_axis)
    short_axis = 2 * min(a_axis, b_axis)

    # Rotate center back to original coords (same matrix as MATLAB)
    R = np.array([[cphi, sphi], [-sphi, cphi]])
    X0_in, Y0_in = (R @ np.array([X0, Y0])).tolist()

    return {
        "a": a_axis,
        "b": b_axis,
        "phi": phi,
        "X0": X0,
        "Y0": Y0,
        "X0_in": X0_in,
        "Y0_in": Y0_in,
        "long_axis": long_axis,
        "short_axis": short_axis,
        "status": "",
    }
