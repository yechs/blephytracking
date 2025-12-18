import numpy as np
import warnings

def fit_ellipse(x, y, axis_handle=None):
    """
    Finds the best fit to an ellipse for the given set of points.

    Parameters:
    - x, y: Arrays of coordinates.
    - axis_handle: Placeholder for compatibility (plotting not implemented here).

    Returns:
    - dictionary containing ellipse parameters.
    """
    orientation_tolerance = 1e-7

    # Prepare vectors
    x = np.array(x).flatten()
    y = np.array(y).flatten()

    mean_x = np.mean(x)
    mean_y = np.mean(y)

    # MATLAB code comments out mean subtraction, so we use raw x, y
    # X = [x.^2, x.*y, y.^2, x, y ]
    X = np.column_stack([x**2, x*y, y**2, x, y])

    # Estimate parameters: a = sum(X)/(X'*X)
    # Equation: a * (X.T @ X) = sum(X) -> (X.T @ X).T * a.T = sum(X).T
    # Since X.T @ X is symmetric: (X.T @ X) * a.T = sum(X).T
    try:
        lhs = X.T @ X
        rhs = np.sum(X, axis=0)
        a_vec = np.linalg.solve(lhs, rhs)
    except np.linalg.LinAlgError:
        warnings.warn('Matrix inversion failed in fit_ellipse')
        return None

    # Extract parameters
    a, b, c, d, e = a_vec

    # Remove orientation
    if min(abs(b/a), abs(b/c)) > orientation_tolerance:
        orientation_rad = 0.5 * np.arctan(b / (c - a))
        cos_phi = np.cos(orientation_rad)
        sin_phi = np.sin(orientation_rad)

        # Rotated coefficients
        at = a*cos_phi**2 - b*cos_phi*sin_phi + c*sin_phi**2
        bt = 0
        ct = a*sin_phi**2 + b*cos_phi*sin_phi + c*cos_phi**2
        dt = d*cos_phi - e*sin_phi
        et = d*sin_phi + e*cos_phi

        # Update mean (though mean wasn't removed in original active code,
        # this tracks the coordinate rotation)
        mean_xt = cos_phi*mean_x - sin_phi*mean_y
        mean_yt = sin_phi*mean_x + cos_phi*mean_y

        a, b, c, d, e = at, bt, ct, dt, et
        mean_x, mean_y = mean_xt, mean_yt
    else:
        orientation_rad = 0
        cos_phi = np.cos(orientation_rad)
        sin_phi = np.sin(orientation_rad)

    # Check if ellipse
    test = a * c
    status = ''
    if test > 0:
        status = ''
    elif test == 0:
        status = 'Parabola found'
        warnings.warn('fit_ellipse: Did not locate an ellipse (Parabola)')
    else:
        status = 'Hyperbola found'
        warnings.warn('fit_ellipse: Did not locate an ellipse (Hyperbola)')

    if test > 0:
        # Make sure coefficients are positive
        if a < 0:
            a, c, d, e = -a, -c, -d, -e

        # Final parameters
        # X0 = -d / (2*a)
        # Y0 = -e / (2*c)
        X0 = -d / (2*a)
        Y0 = -e / (2*c)

        F = 1 + (d**2)/(4*a) + (e**2)/(4*c)

        a_axis = np.sqrt(F / a)
        b_axis = np.sqrt(F / c)

        long_axis = 2 * max(a_axis, b_axis)
        short_axis = 2 * min(a_axis, b_axis)

        # Rotate center back
        R = np.array([[cos_phi, sin_phi], [-sin_phi, cos_phi]])
        P_in = R @ np.array([X0, Y0])
        X0_in = P_in[0]
        Y0_in = P_in[1]

        return {
            'a': a_axis,
            'b': b_axis,
            'phi': orientation_rad,
            'X0': X0,
            'Y0': Y0,
            'X0_in': X0_in,
            'Y0_in': Y0_in,
            'long_axis': long_axis,
            'short_axis': short_axis,
            'status': status
        }
    else:
        return {
            'a': None, 'b': None, 'phi': None,
            'X0': None, 'Y0': None,
            'X0_in': None, 'Y0_in': None,
            'long_axis': None, 'short_axis': None,
            'status': status
        }
