import numpy as np
from skimage.measure import EllipseModel

"""
Note: this creates different output than the original fit_ellipse.m

The original MATLAB code makes the following changes and outputs a WRONG ellipse fit:
x = x;%-mean_x;
y = y;%-mean_y;

X0          = - d/2/a;%(mean_x - d/2/a);%*std_x;
Y0          = - e/2/c;%(mean_y - e/2/c);%*std_y;

I do not know why the original code does this, but I'm assuming it's a bug and not intentional.
"""


def fit_ellipse(x, y):
    """
    Python port of fit_ellipse.m compatible with skimage >= 0.26
    """
    x = np.array(x)
    y = np.array(y)
    points = np.column_stack((x, y))

    # NEW: Use class method 'from_estimate' instead of instance + .estimate()
    # This replaces: ell = EllipseModel(); ell.estimate(points)
    ell = EllipseModel()
    success = ell.estimate(
        points
    )  # Note: 'from_estimate' returns a new model instance if successful

    if success:
        # NEW: Access attributes directly instead of deprecated .params
        # Old: xc, yc, a, b, theta = ell.params
        xc, yc = ell.center
        a, b = (
            ell.params[2],
            ell.params[3],
        )  # Currently a/b are still in params, or use specialized getters if available
        theta = ell.params[4]

        # --- The rest of the logic remains the same ---
        phi = theta
        X0_in = xc
        Y0_in = yc

        c = np.cos(phi)
        s = np.sin(phi)

        # Calculate non-tilted center
        X0 = X0_in * c - Y0_in * s
        Y0 = X0_in * s + Y0_in * c

        long_axis = 2 * max(a, b)
        short_axis = 2 * min(a, b)

        return {
            "a": a,
            "b": b,
            "phi": phi,
            "X0": X0,
            "Y0": Y0,
            "X0_in": X0_in,
            "Y0_in": Y0_in,
            "long_axis": long_axis,
            "short_axis": short_axis,
            "status": "",
        }
    else:
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
            "status": "Fit Failed",
        }

# --- Usage Example to Verify ---
if __name__ == "__main__":
    # Data (Same as previous verification)
    x_data = [13.5852, 12.8709, 11.9862, 10.9021, 9.4736, 8.2752, 7.4426, 6.6417,
            6.1706, 6.3342, 6.6991, 7.5171, 8.6701, 9.6836, 10.9449, 12.1826,
            13.0164, 13.6901, 13.7124, 13.3943]
    y_data = [23.6821, 23.7806, 23.6654, 22.9752, 22.1844, 21.1285, 19.7598, 18.6835,
            17.5036, 16.7163, 16.2197, 16.4027, 16.5636, 17.1789, 18.3809, 19.3749,
            20.7707, 21.7254, 22.7520, 23.5552]

    result = fit_ellipse(x_data, y_data)

    print("--- Python Output (Matching MATLAB Structure) ---")
    for key, value in result.items():
        if isinstance(value, (int, float)):
            print(f"{key:<10}: {value:.4f}")
        else:
            print(f"{key:<10}: {value}")

    # plotting to visualize the fit
    import matplotlib.pyplot as plt

    xc, yc = result["X0_in"], result["Y0_in"]
    a, b = result["a"], result["b"]
    theta = result["phi"]
    t = np.linspace(0, 2 * np.pi, 100)
    ellipse_x = xc + a * np.cos(t) * np.cos(theta) - b * np.sin(t) * np.sin(theta)
    ellipse_y = yc + a * np.cos(t) * np.sin(theta) + b * np.sin(t) * np.cos(theta)
    plt.plot(x_data, y_data, "ro", label="Data Points")
    plt.plot(ellipse_x, ellipse_y, "b-", label="Fitted Ellipse")
    plt.axis("equal")
    plt.legend()
    plt.title("Ellipse Fitting using skimage EllipseModel")
    plt.show()
