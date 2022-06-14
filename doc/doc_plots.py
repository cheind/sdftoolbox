import matplotlib.pyplot as plt
import matplotlib.colors
import numpy as np

import sdftoolbox
from sdftoolbox.roots import directional_newton_roots, bisect_roots


def plot_frames():
    fig, ax = sdftoolbox.plotting.create_figure("ortho")

    ijk = np.stack(np.meshgrid(range(3), range(3), range(3), indexing="ij"), -1)
    colors = np.ones((3, 3, 3, 4))
    colors[:] = matplotlib.colors.to_rgba(next(ax._get_lines.prop_cycler)["color"])
    colors[:-1, :-1, :-1] = matplotlib.colors.to_rgba(
        next(ax._get_lines.prop_cycler)["color"]
    )

    ax.scatter(
        ijk[..., 0],
        ijk[..., 1],
        ijk[..., 2],
        color=colors.reshape(-1, 4),
    )
    ax.plot(
        [ijk[0, 0, 0, 0], ijk[1, 0, 0, 0]],
        [ijk[0, 0, 0, 1], ijk[1, 0, 0, 1]],
        [ijk[0, 0, 0, 2], ijk[1, 0, 0, 2]],
    )

    ax.plot([0, 2], [0, 0], [0, 0], c="k", lw=0.5, linestyle="-")
    ax.plot([0, 0], [0, 2], [0, 0], c="k", lw=0.5, linestyle="-")
    ax.plot([0, 0], [0, 0], [0, 2], c="k", lw=0.5, linestyle="-")
    ax.plot([0, 2], [0, 0], [2, 2], c="k", lw=0.5, linestyle="-")
    ax.plot([0, 0], [0, 2], [2, 2], c="k", lw=0.5, linestyle="-")
    ax.plot([2, 2], [0, 0], [0, 2], c="k", lw=0.5, linestyle="-")
    ax.plot([2, 2], [0, 2], [2, 2], c="k", lw=0.5, linestyle="-")
    ax.plot([0, 0], [2, 2], [0, 2], c="k", lw=0.5, linestyle="-")
    ax.plot([0, 2], [2, 2], [2, 2], c="k", lw=0.5, linestyle="-")
    ax.plot([0, 2], [2, 2], [0, 0], c="k", lw=0.5, linestyle="--")
    ax.plot([2, 2], [0, 2], [0, 0], c="k", lw=0.5, linestyle="--")
    ax.plot([2, 2], [2, 2], [0, 2], c="k", lw=0.5, linestyle="--")

    ax.quiver(0, 0, 0, 1, 0, 0, length=1.0, arrow_length_ratio=0.1, color="red")
    ax.text(0.7, 0.1, 0.0, "i", color="k")

    ax.quiver(0, 0, 0, 0, 1, 0, length=1.0, arrow_length_ratio=0.1, color="green")
    ax.text(0.1, 0.7, 0.0, "j", color="k")

    ax.quiver(0, 0, 0, 0, 0, 1, length=1.0, arrow_length_ratio=0.1, color="blue")
    ax.text(0.05, 0.0, 0.7, "k", color="k")
    sdftoolbox.plotting.setup_axes(
        ax, (-0.5, -0.5, -0.5), (2.5, 2.5, 2.5), azimuth=-121, elevation=32
    )
    plt.tight_layout()
    fig.savefig("doc/frames.svg")
    plt.close(fig)


def plot_edges():
    fig, ax = sdftoolbox.plotting.create_figure("ortho")

    ijk = np.stack(np.meshgrid(range(3), range(3), range(3), indexing="ij"), -1)

    ax.scatter(ijk[..., 0], ijk[..., 1], ijk[..., 2], label="sample coords")

    ax.quiver(1, 0, 0, 1, 0, 0, length=1.0, arrow_length_ratio=0.1, color="purple")
    ax.text(1.5, -0.1, -0.1, "(1,0,0,0)", color="k")

    ax.quiver(0, 1, 1, 0, 0, 1, length=1.0, arrow_length_ratio=0.1, color="purple")
    ax.text(0, 2.1, 1.5, "(0,1,1,2)", color="k")

    ax.quiver(2, 0, 2, 0, 1, 0, length=1.0, arrow_length_ratio=0.1, color="purple")
    ax.text(2, 0.5, 2.1, "(2,0,2,1)", color="k")

    sdftoolbox.plotting.setup_axes(
        ax, (-0.5, -0.5, -0.5), (2.5, 2.5, 2.5), azimuth=-110, elevation=38
    )
    plt.legend()
    plt.tight_layout()
    fig.savefig("doc/edges.svg")
    plt.close(fig)


def plot_edge_strategies():
    def compute_linear_isect(node, edge):
        edge_sdf = node.sample(edge)
        tlin = sdftoolbox.LinearEdgeStrategy.compute_linear_roots(
            edge_sdf[0:1], edge_sdf[1:2]
        ).squeeze()
        xlin = (1 - tlin) * edge[0] + tlin * edge[1]
        return xlin

    def compute_newton_isect(node, edge):
        dir = edge[1] - edge[0]
        dir = dir / np.linalg.norm(dir)
        x = compute_linear_isect(node, edge)
        x = directional_newton_roots(node, x[None, :], dir[None, :])
        return x.squeeze()

    def compute_bisect_isect(node, edge):
        x = compute_linear_isect(node, edge)
        x = bisect_roots(node, edge[0:1], edge[1:2], x[None, :], max_steps=20)
        return x.squeeze()

    def plot_edge(ax, edge, isect):
        ax.plot(edge[:, 0], edge[:, 1], color="k", linewidth=0.5)
        ax.scatter(edge[:, 0], edge[:, 1], color="k", s=20, zorder=3)
        ax.scatter(
            isect[0],
            isect[1],
            s=20,
            marker="o",
            zorder=3,
            facecolors="none",
            edgecolors="r",
        )

    # Sphere
    fig, axs = plt.subplots(1, 3, figsize=plt.figaspect(0.3333))
    sphere = sdftoolbox.sdfs.Sphere.create(radius=1.0)
    sphere_grid = sdftoolbox.sdfs.Grid(
        (100, 100, 1), min_corner=(0.0, 0.0, 0.0), max_corner=(1.1, 1.1, 0.0)
    )
    xyz = sphere_grid.xyz
    sdf = sphere.sample(xyz)

    edge1 = np.array([[0.622, 0.674, 0], [0.810, 0.753, 0]])
    edge2 = np.array([[0.5, 0.2, 0], [0.5, 0.995, 0]])

    cs = axs[0].contour(xyz[..., 0, 0], xyz[..., 0, 1], sdf[..., 0])
    axs[0].clabel(cs, inline=True, fontsize=10)
    cs = axs[1].contour(xyz[..., 0, 0], xyz[..., 0, 1], sdf[..., 0])
    axs[1].clabel(cs, inline=True, fontsize=10)
    cs = axs[2].contour(xyz[..., 0, 0], xyz[..., 0, 1], sdf[..., 0])
    axs[2].clabel(cs, inline=True, fontsize=10)
    axs[0].set_title("Linear")
    axs[1].set_title("Directional Newton")
    axs[2].set_title("Bisection")
    plot_edge(axs[0], edge1, compute_linear_isect(sphere, edge1))
    plot_edge(axs[0], edge2, compute_linear_isect(sphere, edge2))
    plot_edge(axs[1], edge1, compute_newton_isect(sphere, edge1))
    plot_edge(axs[1], edge2, compute_newton_isect(sphere, edge2))
    plot_edge(axs[2], edge1, compute_bisect_isect(sphere, edge1))
    plot_edge(axs[2], edge2, compute_bisect_isect(sphere, edge2))
    plt.tight_layout()
    fig.savefig("doc/edge_strategies_sphere.svg")
    plt.close(fig)

    # Box
    fig, axs = plt.subplots(1, 3, figsize=plt.figaspect(0.3333))
    box = sdftoolbox.sdfs.Box()
    box_grid = sdftoolbox.sdfs.Grid(
        (100, 100, 1), min_corner=(0.0, 0.0, 0.0), max_corner=(1.1, 1.1, 0.0)
    )
    xyz = box_grid.xyz
    sdf = box.sample(xyz)

    edge1 = np.array([[0.448, 0.039, 0], [0.448, 0.649, 0]])
    edge2 = np.array([[0.283, 0.259, 0], [0.866, 0.757, 0]])

    cs = axs[0].contour(xyz[..., 0, 0], xyz[..., 0, 1], sdf[..., 0])
    axs[0].clabel(cs, inline=True, fontsize=10)
    cs = axs[1].contour(xyz[..., 0, 0], xyz[..., 0, 1], sdf[..., 0])
    axs[1].clabel(cs, inline=True, fontsize=10)
    cs = axs[2].contour(xyz[..., 0, 0], xyz[..., 0, 1], sdf[..., 0])
    axs[2].clabel(cs, inline=True, fontsize=10)
    axs[0].set_title("Linear")
    axs[1].set_title("Directional Newton")
    axs[2].set_title("Bisection")
    plot_edge(axs[0], edge1, compute_linear_isect(box, edge1))
    plot_edge(axs[0], edge2, compute_linear_isect(box, edge2))
    plot_edge(axs[1], edge1, compute_newton_isect(box, edge1))
    plot_edge(axs[1], edge2, compute_newton_isect(box, edge2))
    plot_edge(axs[2], edge1, compute_bisect_isect(box, edge1))
    plot_edge(axs[2], edge2, compute_bisect_isect(box, edge2))
    plt.tight_layout()
    fig.savefig("doc/edge_strategies_box.svg")
    plt.close(fig)


def plot_vertex_strategies():
    # Box in canonical orientation

    def plot_boxes(boxes, grid):
        fig = plt.figure(figsize=plt.figaspect(0.3333))
        ax0 = fig.add_subplot(
            1, 3, 1, projection="3d", proj_type="persp", computed_zorder=False
        )
        ax1 = fig.add_subplot(
            1, 3, 2, projection="3d", proj_type="persp", computed_zorder=False
        )
        ax2 = fig.add_subplot(
            1, 3, 3, projection="3d", proj_type="persp", computed_zorder=False
        )

        # for the contour plot
        minc = grid.min_corner.copy()
        minc[2] = 0
        maxc = grid.max_corner.copy()
        maxc[2] = 0
        xyz = sdftoolbox.sdfs.Grid((100, 100, 1), min_corner=minc, max_corner=maxc).xyz
        sdf = boxes.sample(xyz)
        cs = ax0.contour(
            xyz[..., 0, 0],
            xyz[..., 0, 1],
            sdf[..., 0],
            zdir="z",
            offset=0,
            levels=[0],
            colors="purple",
        )
        cs = ax1.contour(
            xyz[..., 0, 0],
            xyz[..., 0, 1],
            sdf[..., 0],
            zdir="z",
            offset=0,
            levels=[0],
            colors="purple",
        )
        cs = ax2.contour(
            xyz[..., 0, 0],
            xyz[..., 0, 1],
            sdf[..., 0],
            zdir="z",
            offset=0,
            levels=[0],
            colors="purple",
        )

        verts0, faces0 = sdftoolbox.dual_isosurface(
            boxes, grid, vertex_strategy=sdftoolbox.MidpointVertexStrategy()
        )
        verts1, faces1 = sdftoolbox.dual_isosurface(
            boxes, grid, vertex_strategy=sdftoolbox.NaiveSurfaceNetVertexStrategy()
        )
        verts2, faces2 = sdftoolbox.dual_isosurface(
            boxes,
            grid,
            vertex_strategy=sdftoolbox.DualContouringVertexStrategy(),
            edge_strategy=sdftoolbox.BisectionEdgeStrategy(),
        )
        sdftoolbox.plotting.setup_axes(ax0, grid.min_corner, grid.max_corner)
        sdftoolbox.plotting.setup_axes(ax1, grid.min_corner, grid.max_corner)
        sdftoolbox.plotting.setup_axes(ax2, grid.min_corner, grid.max_corner)
        ax0.set_title("sdftoolbox.MidpointVertexStrategy")
        ax1.set_title("sdftoolbox.NaiveSurfaceNetVertexStrategy")
        ax2.set_title("sdftoolbox.DualContouringVertexStrategy")
        sdftoolbox.plotting.plot_mesh(ax0, verts0, faces0)
        sdftoolbox.plotting.plot_mesh(ax1, verts1, faces1)
        sdftoolbox.plotting.plot_mesh(ax2, verts2, faces2)
        return fig, (ax0, ax1, ax2)

    grid = sdftoolbox.sdfs.Grid(
        (10, 10, 10), min_corner=(-1.1, -1.1, -1.1), max_corner=(1.1, 1.1, 1.1)
    )
    # Canonical aligned boxes
    boxes = sdftoolbox.sdfs.Union(
        [
            sdftoolbox.sdfs.Box().transform(trans=(0.5, 0.5, 0.5)),
            sdftoolbox.sdfs.Box(),
        ]
    ).transform(trans=(-0.25, -0.25, -0.25))
    fig, axs = plot_boxes(boxes, grid)
    sdftoolbox.plotting.generate_rotation_gif(
        "doc/vertex_strategies_aligned_box.gif", fig, axs, total_time=10
    )
    plt.close(fig)

    # Rotate boxes
    boxes = sdftoolbox.sdfs.Union(
        [
            sdftoolbox.sdfs.Box().transform(trans=(0.5, 0.5, 0.5)),
            sdftoolbox.sdfs.Box(),
        ]
    ).transform(trans=(-0.25, -0.25, -0.25), rot=(1, 1, 1, np.pi / 4))
    fig, axs = plot_boxes(boxes, grid)
    sdftoolbox.plotting.generate_rotation_gif(
        "doc/vertex_strategies_rot_box.gif", fig, axs, total_time=10
    )
    plt.close(fig)


if __name__ == "__main__":

    # plot_frames()
    # plot_edges()
    # plot_edge_strategies()
    plot_vertex_strategies()
