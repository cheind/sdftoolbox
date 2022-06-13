import matplotlib.pyplot as plt
import matplotlib.colors
import numpy as np

import sdftoolbox


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
    # ax.scatter(
    #     ijk[:-1, :-1, :-1, 0],
    #     ijk[:-1, :-1, :-1, 1],
    #     ijk[:-1, :-1, :-1, 2],
    #     label="voxel coords",
    #     alpha=1,
    #     marker="x",
    # )
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


if __name__ == "__main__":

    plot_frames()
    plot_edges()
