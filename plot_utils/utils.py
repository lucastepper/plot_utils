import string
from typing import Iterable
import matplotlib as mpl
import numpy as np
import matplotlib.pyplot as plt


def get_size(width: float, scale: float = 1, subplots: tuple[int] = (1, 1), ratio: float = None):
    """Return figure size for given width and scale.  The figure size is
    adjusted to the subplot amount. The figure height is calculated based
    on the golden ration. If ratio is given, it is scaled down.
    Args:
        width: width of the figure in points.
        scale: scale for figure size.
        subplots: Argmuent to plt.subplots.
        ratio: ratio between figure height and width.
            default: None, uses golden ratio.
    Returns:
        figsize (tuple[float, float]): figure size in inches.
    """
    # Width of figure (in pts)
    fig_width_pt = width * scale
    # Convert from pt to inches
    inches_per_pt = 1 / 72.27
    # Golden ratio to set aesthetic figure height
    if not ratio:
        ratio = (5**0.5 - 1) / 2
    # Figure width in inches
    fig_width_in = fig_width_pt * inches_per_pt
    # Figure height in inches
    fig_height_in = fig_width_in * ratio * (subplots[0] / subplots[1])
    return (fig_width_in, fig_height_in)


def get_figure(
    subplot1: int,
    subplot2: int,
    wspace: float = 0.3,
    hspace: float = 0.4,
    scale: float = 1,
    sharex: bool = False,
    sharey: bool = False,
    ratio: tuple[float, float] = None,
    width: int = 520,
):
    """Return plt figure and axes object for given subplot dim, scale
    between subplots and overall scale. Uses predefined LaTex width.
    Arguments
    subplot1: Number of subplots in horizontal dir
    subplot2: Number of subplots in vertical dir
    wspace: space between subplots in horizontal direction
    hspace: space between subplots in vertical direction
    scale: scale whole plot down by
    ration: ratio between width and height
    width: width in points. A4 paper is about 460
    """
    fig, axes = plt.subplots(
        subplot1,
        subplot2,
        sharex=sharex,
        sharey=sharey,
        figsize=get_size(width, scale=scale, subplots=(subplot1, subplot2), ratio=ratio),
    )
    fig.subplots_adjust(wspace=wspace, hspace=hspace)
    return fig, axes


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


def to_int(x):
    """Convert number to integer if x % 1 == 0"""
    if int(x) == x:
        return int(x)
    else:
        return x


def add_letters(axes, letters: Iterable[str] = string.ascii_uppercase, **kwargs):
    """Add capital letters to the each subplot for identification it its caption.
    By default, each letter is at the top right corner of the subplot.
    To overwrite the position of subplot i, give kwarg pos_i=(shift, height).
    To overwrite default, give kwarg pos=(shift, height).
    Alignment can be changed by kwarg va='top'/'bottom'
        or ha='left'/'right'.
    Arguments:
        letters (Iterable[str]): The letters to use. default: string.ascii_uppercase
    """
    pos = kwargs.get("pos", (0.9, 0.9))
    va = kwargs.get("va", "top")
    ha = kwargs.get("ha", "right")
    for i, (ax, letter) in enumerate(zip(axes, letters)):
        shift, height = kwargs.get(f"pos_{i}", pos)
        ax.text(shift, height, letter, transform=ax.transAxes, fontweight="bold", va=va, ha=ha)


def set_scientific_format(
    axis: mpl.axes.Axes, yaxis: bool = True, xaxis: bool = True, scilimits: tuple[int] = (0, 0)
):
    """Change the notation style of the given axis to scientific notation.
    Surpresses the error that is thrown when the axis is logarithmic.
    for both xaxis=True and yaxis=True, try both, again ignoring error.
    Arguments:
        axis: The axis to change.
        yaxis: Whether to change the yaxis. default: True
        xaxis: Whether to change the xaxis. default: True
        scilimits: The scilimits to use. default: (0, 0)
    """
    for use_ax, ax in zip((xaxis, yaxis), ("x", "y")):
        if use_ax:
            try:
                axis.ticklabel_format(axis=ax, style="sci", scilimits=(0, 0))
            except AttributeError:
                pass
