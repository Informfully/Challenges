import json

import yaml


def yaml_to_dict(yaml_path: str) -> dict:
    with open(yaml_path, 'r') as file:
        return yaml.safe_load(file)
    

def json_to_dict(json_path: str) -> dict:
    with open(json_path, 'r') as file:
        return json.load(file)
