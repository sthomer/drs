from pathlib import Path
import yaml

def _merge_dicts(base: dict, override: dict) -> dict:
    result = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _merge_dicts(result[key], value)
        else:
            result[key] = value
    return result


def load_config(path: str) -> dict:
    with open(Path(path)) as file:
        config = yaml.safe_load(file)

    parent = config.pop("inherits", None)
    if parent:
        parent_config = load_config(parent)
        config = _merge_dicts(parent_config, config)

    return config
