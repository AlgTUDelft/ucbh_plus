from typing import Dict, List, Union
import numpy as np


def fetch_stat(
        data: List[Dict[str, Union[str, int, float]]],
        stat: str,
        episodes: int,
        trials: int,
) -> Dict[str, np.ndarray]:
    """
    reshapes the data for each agent into a numpy array and fills it with a single stat for plotting
    :param data: a list of dictionaries with the experiment's data
    :param stat: a stat to use in filling the arrays; all other stats will be discarded
    :param episodes: number of episodes
    :param trials: number of trials
    :return: a dictionary of numpy arrays with a given stat for each agent
    """
    result = {}
    for item in data:
        method = item['method']
        if method not in result:
            result[method] = np.full((trials, episodes), np.inf)
        episode = item.get('episode')
        trial = item.get('trial')
        if episode is not None and trial is not None:
            result[method][trial, episode] = item.get(stat)

    return result
