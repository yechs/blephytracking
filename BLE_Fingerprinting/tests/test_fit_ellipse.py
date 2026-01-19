import numpy as np

from fit_ellipse import fit_ellipse


def test_fit_ellipse_recovers_center():
    rng = np.random.default_rng(0)
    a = 3.0
    b = 2.0
    phi = 0.3
    x0 = 1.5
    y0 = -2.0

    theta = np.linspace(0, 2 * np.pi, 200)
    x = a * np.cos(theta)
    y = b * np.sin(theta)

    rot = np.array([[np.cos(phi), -np.sin(phi)], [np.sin(phi), np.cos(phi)]])
    pts = rot @ np.vstack([x, y])
    x_rot = pts[0, :] + x0
    y_rot = pts[1, :] + y0

    x_rot += rng.normal(scale=1e-3, size=x_rot.shape)
    y_rot += rng.normal(scale=1e-3, size=y_rot.shape)

    ellipse = fit_ellipse(x_rot, y_rot)

    assert ellipse.status == ""
    np.testing.assert_allclose([ellipse.X0, ellipse.Y0], [x0, y0], atol=5e-2)
