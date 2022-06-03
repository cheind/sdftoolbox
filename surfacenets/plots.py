import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


def create_figure():
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    return fig, ax


def create_split_figure(sync: bool = True):
    fig = plt.figure(figsize=plt.figaspect(0.5))
    ax0 = fig.add_subplot(1, 2, 1, projection="3d")
    ax1 = fig.add_subplot(1, 2, 2, projection="3d")

    sync_pending = False
    sync_dir = [None, None]

    def sync_views(a, b):
        b.view_init(elev=a.elev, azim=a.azim)
        b.set_xlim3d(a.get_xlim3d())
        b.set_ylim3d(a.get_ylim3d())
        b.set_zlim3d(a.get_zlim3d())

    def on_press(event):
        nonlocal sync_pending, sync_dir
        inaxes = event.inaxes in [ax0, ax1]
        if inaxes:
            sync_pending = True
            sync_dir = [ax0, ax1] if event.inaxes == ax0 else [ax1, ax0]

    def on_release(event):
        nonlocal sync_pending

        if sync_pending:
            sync_views(*sync_dir)
            sync_pending = False
            fig.canvas.draw_idle()

    if sync:
        fig.canvas.mpl_connect("button_press_event", on_press)
        fig.canvas.mpl_connect("button_release_event", on_release)

    return fig, (ax0, ax1)


def setup_axes(
    ax,
    min_corner,
    max_corner,
    azimuth: float = -139,
    elevation: float = 35,
    num_grid: int = 3,
):

    ax.set_xlim(min_corner[0], max_corner[0])
    ax.set_ylim(min_corner[1], max_corner[1])
    ax.set_zlim(min_corner[2], max_corner[2])
    ax.xaxis.set_major_locator(MaxNLocator(num_grid))
    ax.yaxis.set_major_locator(MaxNLocator(num_grid))
    ax.zaxis.set_major_locator(MaxNLocator(num_grid))
    ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_box_aspect(
        (
            max_corner[0] - min_corner[0],
            max_corner[1] - min_corner[1],
            max_corner[2] - min_corner[2],
        )
    )
    ax.view_init(elev=elevation, azim=azimuth)
