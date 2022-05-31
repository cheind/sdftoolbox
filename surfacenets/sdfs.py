import numpy as np
import abc

from . import maths


class SDF(abc.ABC):
    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.value(x)

    @abc.abstractmethod
    def value(self, x: np.ndarray) -> np.ndarray:
        ...

    def merge(self, *others: list["SDF"], alpha: float = np.inf) -> "Union":
        return Union([self] + list(others), alpha=alpha)

    def intersect(self, *others: list["SDF"], alpha: float = np.inf) -> "Intersection":
        return Intersection([self] + list(others), alpha=alpha)

    def subtract(self, *others: list["SDF"], alpha: float = np.inf) -> "Difference":
        return Difference([self] + list(others), alpha=alpha)


class Sphere(SDF):
    def __init__(self, center: np.ndarray, radius: float) -> None:
        self.center = np.asarray(center).reshape(3)
        self.radius = radius

    def value(self, x: np.ndarray) -> np.ndarray:
        x = np.atleast_2d(x)
        d2 = np.square(x - self.center[..., :]).sum(-1)
        return d2 - self.radius**2


class Union(SDF):
    def __init__(self, sdfs: list[SDF], alpha: float = np.inf) -> None:
        if len(sdfs) == 0:
            raise ValueError("Need at least one SDF")
        self.children = sdfs
        self.alpha = alpha

    def value(self, x: np.ndarray) -> np.ndarray:
        values = np.stack([c.value(x) for c in self.children], 0)
        # min = -max(-values)
        return -maths.generalized_max(-values, 0, alpha=self.alpha)


class Intersection(SDF):
    def __init__(self, sdfs: list[SDF], alpha: float = np.inf) -> None:
        if len(sdfs) == 0:
            raise ValueError("Need at least one SDF")
        self.children = sdfs
        self.alpha = alpha

    def value(self, x: np.ndarray) -> np.ndarray:
        values = np.stack([c.value(x) for c in self.children], 0)
        return maths.generalized_max(values, 0, alpha=self.alpha)


class Difference(SDF):
    def __init__(self, sdfs: list[SDF], alpha: float = np.inf) -> None:
        if len(sdfs) == 0:
            raise ValueError("Need at least one SDF")
        self.children = sdfs
        self.alpha = alpha

    def value(self, x: np.ndarray) -> np.ndarray:
        values = np.stack([c.value(x) for c in self.children], 0)
        values[1:] *= -1
        return maths.generalized_max(values, 0, alpha=self.alpha)


if __name__ == "__main__":
    # https://0fps.net/2012/07/12/smooth-voxel-terrain-part-2/
    from skimage.measure import marching_cubes
    import matplotlib.pyplot as plt
    import time

    res = (40, 40, 40)
    min_corner = np.array([-2.0] * 3)
    max_corner = np.array([2.0] * 3)

    ranges = [
        np.linspace(min_corner[0], max_corner[0], res[0]),
        np.linspace(min_corner[1], max_corner[1], res[1]),
        np.linspace(min_corner[2], max_corner[2], res[2]),
    ]

    X, Y, Z = np.meshgrid(*ranges)
    xyz = np.stack((X, Y, Z), -1)
    spacing = (
        ranges[0][1] - ranges[0][0],
        ranges[1][1] - ranges[1][0],
        ranges[2][1] - ranges[2][0],
    )

    s1 = Sphere([0.0, 0.0, 0.0], 1.0)
    s2 = Sphere([1.0, 0.0, 0.0], 1.0)
    sdf = s1.subtract(s2, alpha=0.5)
    t0 = time.perf_counter()
    values = sdf(xyz)
    print("Eval SDF took", time.perf_counter() - t0, "secs")

    verts, faces, normals, _ = marching_cubes(values, 0.0, spacing=spacing)
    verts += min_corner[None, :]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_trisurf(
        verts[:, 0],
        verts[:, 1],
        faces,
        verts[:, 2],
        cmap="Spectral",
        antialiased=True,
        linewidth=0
        # edgecolor="white",
    )
    ax.set_xlim(min_corner[0], max_corner[0])
    ax.set_ylim(min_corner[1], max_corner[1])
    ax.set_zlim(min_corner[2], max_corner[2])
    ax.set_box_aspect(
        (
            max_corner[0] - min_corner[0],
            max_corner[1] - min_corner[1],
            max_corner[2] - min_corner[2],
        )
    )

    plt.show()

    # fig = go.Figure(
    #     data=go.Isosurface(
    #         x=xyz[..., 0].flatten(),
    #         y=xyz[..., 1].flatten(),
    #         z=xyz[..., 2].flatten(),
    #         value=values.flatten(),
    #         isomin=0,
    #         isomax=0,
    #         # caps=dict(x_show=False, y_show=False),
    #     )
    # )
    # fig.show()
