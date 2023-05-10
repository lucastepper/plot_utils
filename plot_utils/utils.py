import numpy as np
import matplotlib.pyplot as plt


def get_log_func(base: float):
    """Get the logarithm function for a given base,
    numpy only has log naturalis, log2 and log10.
    """
    if base == 2:
        return np.log2
    elif base == 10:
        return np.log10
    elif np.abs(base - np.e) < 1e-2:
        if not np.abs(base - np.e) < 1e-14:
            print("Warning: Base is not exactly e, but close, might give artifacts.")
        return np.log
    else:
        raise ValueError(f"Base must be 2, 10 or 2.718, but is {base}")


def subsample_logspace(data: np.ndarray, n: int, dt: float, base: float = 2.0):
    """Subsample data logarithmically in time.
    Arguments:
        data: The data to subsample.
        n: The number of points to subsample.
        dt: The time step of the data.
        base: The base of the logarithm (2, 10 or 2.718). default: 2
    Returns:
        time: The time points of the subsampled data.
        data: The subsampled data.
    """
    if n >= len(data):
        return np.arange(len(data)) * dt, data
    end = get_log_func(base)(len(data))
    idxs = np.unique(np.logspace(0, end, n, base=base, dtype=int)) - 1
    time = idxs * dt
    return time, data[idxs]


def get_time_logspace(tstart: float, tend: float, n: int, dt: float, base: float = 2.0):
    """Get a logarithmically spaced time array.
    WARNING, this does not give exactly the same time array that subsample_logspace
    gives, for some rounding reasons, at least when tstart is 0.
    Arguments:
        tstart: The start time.
        tend: The end time.
        dt: The time step of the data.
        base: The base of the logarithm. default: 2
    Returns:
        time: The time points of the subsampled data.
    """
    start_was_zero = False
    if tstart == 0.0:
        start_was_zero = True
        tstart = dt
        n -= 1
    start = get_log_func(base)(tstart)
    end = get_log_func(base)(tend)
    time = np.logspace(start, end, n, base=base) / dt
    # convert to int and filter unique
    time = np.unique(time.astype(int))
    if start_was_zero:
        time = np.insert(time, 0, 0)
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
