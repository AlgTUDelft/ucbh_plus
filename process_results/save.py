import os
import csv
from typing import Dict, List, Union


def save(path: str, env_name: str, results: List[Dict[str, Union[str, int, float]]]):
    file_name = f'{path}/{env_name}.csv'
    file_exists = os.path.exists(file_name)
    if not file_exists:
        os.makedirs(path, exist_ok=True)
    field_names = results[0].keys()
    with open(file_name, 'a') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=field_names)
        if not file_exists:
            writer.writeheader()
        writer.writerows(results)
