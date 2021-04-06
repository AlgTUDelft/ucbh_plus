import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as st
from typing import Dict, List, Union, Optional


DEFAULT_MOVING_AVERAGE_ORDER = 5        # smoothing parameter
DEFAULT_PLOT_QUANTILES = True           #
DEFAULT_INTERQUANTILE_RANGE = 1.0       #
DEFAULT_COLORS = [                      # TU Delft colors for plotting
    '#666666',                  # gray
    '#E64616',                  # orange
    '#00A6D6',                  # cyan
    '#E1C400',                  # yellow
    '#6D177F',                  # warm purple
    '#A5CA1A',                  # bright green
    '#E21A1A',                  # red
    '#6EBBD5',                  # sky blue
    '#008891',                  # green
    '#1D1C73',                  # purple
    '#6B8689',                  # grey green
]


def _moving_average(a, n=DEFAULT_MOVING_AVERAGE_ORDER):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n


def _plot_data(x, y, q=None, color=None, alpha_fill=0.2, label=None, ma=DEFAULT_MOVING_AVERAGE_ORDER, ax=None):

    ax = ax if ax is not None else plt.gca()
    if color is None:
        color = ax._get_lines.color_cycle.next()

    avg = _moving_average(np.average(y, axis=0), ma)
    line = ax.plot(x, avg, color=color, label=label)[0]
    if len(y) > 1:
        sem = _moving_average(st.sem(y, axis=0), ma)
        y_min = avg - sem
        y_max = avg + sem
        ax.fill_between(x, y_max, y_min, color=color, alpha=alpha_fill)
        if q is not None:
            q = min(q, 1-q)
            upper_quantile = _moving_average(np.quantile(y, 1 - q, axis=0), ma)
            lower_quantile = _moving_average(np.quantile(y, q, axis=0), ma)
            ax.fill_between(x, upper_quantile, lower_quantile, color=color, alpha=alpha_fill)
    return line


def plot(data: Dict[str, np.ndarray],
         title: str,
         colors: List[str] = None,
         ma: int = DEFAULT_MOVING_AVERAGE_ORDER,
         show_q: bool = DEFAULT_PLOT_QUANTILES,
         iqr: float = DEFAULT_INTERQUANTILE_RANGE,
         episodes: Optional[int] = None,
         solution: float = None):

    if colors is None:
        colors = DEFAULT_COLORS
    agent_colors = {}
    agents = list(data.keys())
    if episodes is None:
        episodes = data[agents[0]].shape[0]
    results = data
    ma = max(ma, 1)

    for agent_n in range(len(agents)):
        agent_name = agents[agent_n]
        agent_colors[agent_name] = colors[agent_n % len(colors)]

    x = np.arange(0, episodes - (ma - 1))

    if solution is not None:
        plt.plot(x, [solution] * len(x), color='black', linestyle='--', linewidth=0.75)

    lines = []
    for agent_name in agents:
        q = (1.0 - iqr) / 2 if show_q else None
        l = _plot_data(x, results[agent_name], q, agent_colors[agent_name], label=agent_name, ma=ma)
        lines.append(l)
    plt.legend(loc=4)
    if title is not None:
        plt.title(title)
    plt.show()
