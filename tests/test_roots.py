import numpy as np
from surfacenets.sdfs import Sphere
from surfacenets.roots import directional_newton_roots


def test_newton_root_finding():
    np.random.seed(123)
    s = Sphere()
    x0 = np.random.uniform(-2, 2, size=(20, 3))
    x = directional_newton_roots(s, x0, dirs=None)

    # Result should be on sphere
    np.allclose(np.linalg.norm(x, axis=-1), 1.0)

    # The way the points moved should coincide with the initial
    # gradient direction
    n0 = s.gradient(x0, normalize=True)
    d = (x - x0) / np.linalg.norm(x - x0, axis=-1, keepdims=True)
    np.allclose(np.abs(n0[:, None, :] @ d[..., None]), 0)

    # Another test with fixed directions
    x0 = np.array([[0.1, 0.0, 0.0], [0.0, 0.1, 0.0], [0.0, 0.0, 1.1]])
    d = np.array([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0]])
    x = directional_newton_roots(s, x0, dirs=d)
    np.allclose(x, np.eye(3))
