import numpy as np
import matplotlib.pyplot as plt


def subsample_logspace(data: np.ndarray, n: int, dt: float, base=2):
    """Subsample data logarithmically in time.
    Arguments:
        data: The data to subsample.
        n: The number of points to subsample.
        dt: The time step of the data.
        base: The base of the logarithm (2, 10 or 'e'). default: 2
    Returns:
        time: The time points of the subsampled data.
        data: The subsampled data.
    """
    if n >= len(data):
        return np.arange(len(data)) * dt, data
    if base == 2:
        end = np.log2(len(data))
    elif base == 10:
        end = np.log10(len(data))
    elif base == "e":
        base = np.e
        end = np.log(len(data))
    else:
        raise ValueError(f"Base must be 2, 10 or 'e', but is {base}")
    idxs = np.unique(np.logspace(0, end, n, base=base, dtype=int)) - 1
    time = idxs * dt
    return time, data[idxs]


def get_time_logspace(tstart: int, tend: int, n: int, dt: float, base=2):
    """Get a logarithmically spaced time array.
    Arguments:
        tstart: The start time.
        tend: The end time.
        dt: The time step of the data.
        base: The base of the logarithm. default: 2
    Returns:
        time: The time points of the subsampled data.
    """
    if base == 2:
        start, end = np.log2(tstart), np.log2(tend)
    elif base == 10:
        start, end = np.log10(tstart), np.log10(tend)
    else:
        raise ValueError(f"Base was {base}, but needs to be 2 or 10.")
    time = np.logspace(start, end, n, base=base) / dt
    # convert to int and filter unique
    time = np.unique(time.astype(int))
    return time * dt


def format_axes_space(ax: plt.Axes):
    """Format an axes that is not used for plotting but for the space between
    two plots in horizontal direction. Serves to make such individual wspaces.
    Arguments:
        ax: The axes to format.
    """
    ax.tick_params(
        left=False, right=False, top=False, bottom=False, labelleft=False, labelbottom=False
    )
    ax.spines["bottom"].set_color(None)
    ax.spines["top"].set_color(None)
    ax.spines["left"].set_color(None)
    ax.spines["right"].set_color(None)
    ax.set_zorder(0)
