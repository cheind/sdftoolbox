import numpy as np
from surfacenets.sdfs import Box, Sphere
from surfacenets.roots import directional_newton_roots, bisect_roots


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


def test_newton_single_dir():
    np.random.seed(123)
    s = Sphere()
    x0 = np.random.randn(100, 3) * 1e-2
    x = directional_newton_roots(s, x0, dirs=np.array([1.0, 0.0, 0.0]))
    # Starting from close to center they should fly off to ~ -1/1 in x.
    assert np.allclose(np.linalg.norm(x, axis=-1), 1.0)
    assert np.allclose(
        np.abs(x[:, None, :] @ np.array([[1.0, 0.0, 0.0]])[..., None]).squeeze(-1),
        1.0,
        atol=1e-2,
    )


def test_bisect_root_finding():
    np.random.seed(123)
    s = Sphere()
    dirs = np.random.uniform(-2, 2, size=(20, 3))
    dirs /= np.linalg.norm(dirs, axis=-1, keepdims=True)

    a = np.zeros_like(dirs)
    b = dirs * 2
    x0 = b * np.random.uniform(0, 1, size=(len(dirs), 1))

    # Midpoint bisection
    x = bisect_roots(s, a, b, x0, max_steps=50)
    assert np.allclose(s.sample(x), 0, 1e-5)

    # Less iterations should fail
    x = bisect_roots(s, a, b, x0, max_steps=1)
    assert not np.allclose(s.sample(x), 0, 1e-5)

    # Linear interpolation shoudl speed up though (at least for sphere)
    x = bisect_roots(s, a, b, x0, max_steps=1, linear_interp=True)
    assert np.allclose(s.sample(x), 0, 1e-5)

    # Same thing with a tricky box SDF
    s = Box()
    a = np.array([[0.48, 0.6, 0.0]])
    b = np.array([[0.48, -0.3, 0.0]])
    assert np.allclose(s.sample(a), 0.1)
    assert np.allclose(s.sample(b), -0.02)

    # Standard bisect converges faster as linear interp. is not misleading
    x = bisect_roots(s, a, b, max_steps=12)
    assert np.allclose(x, [[0.48, 0.5, 0.0]], 1e-3)

    # Linear interp. converges slower
    x = bisect_roots(s, a, b, linear_interp=True, max_steps=12)
    assert not np.allclose(x, [[0.48, 0.5, 0.0]], 1e-3)
    assert np.allclose(x, [[0.48, 0.5, 0.0]], 1e-1)
