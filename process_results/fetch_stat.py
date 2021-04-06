from typing import Dict, List, Union
import numpy as np


def fetch_stat(
        data: List[Dict[str, Union[str, int, float]]],
        stat: str,
        episodes: int,
        trials: int,
) -> Dict[str, np.ndarray]:
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
