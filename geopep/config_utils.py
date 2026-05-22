"""Config helpers: resolve relative paths in config.yaml to absolute paths.

Relative paths are interpreted as relative to the project root (the parent
directory of the config file's directory). Example:

    Config file:        <repo>/configs/config.yaml
    Path in YAML:       "pdb/complex"
    Resolves to:        <repo>/pdb/complex

Absolute paths are left untouched.
"""

import os
from typing import Any


# (section, key) pairs whose values are single path strings.
_PATH_FIELDS = [
    ('preprocess', 'complex_directory'),
    ('preprocess', 'interface_directory'),
    ('preprocess', 'output_directory'),
    ('prediction', 'checkpoint_path'),
    ('prediction', 'input_json'),
    ('training', 'checkpoint_dir'),
]

# (section, key) pairs whose values are lists of path strings.
_LIST_PATH_FIELDS = [
    ('data', 'train_json'),
    ('data', 'val_json'),
]


def project_root_from_config(config_path: str) -> str:
    """Return the project root directory.

    Assumes the config file lives in `<project_root>/configs/`.
    """
    return os.path.dirname(os.path.dirname(os.path.abspath(config_path)))


def _resolve_one(value: Any, base_dir: str) -> Any:
    if not isinstance(value, str) or not value:
        return value
    if os.path.isabs(value):
        return value
    return os.path.normpath(os.path.join(base_dir, value))


def resolve_paths(config: dict, config_path: str) -> dict:
    """In-place: convert relative paths in `config` to absolute paths.

    Relative paths are resolved against `<project_root>/`, where project root
    is the parent of the config file's directory.

    Absolute paths are passed through unchanged.
    """
    base = project_root_from_config(config_path)

    for section, key in _PATH_FIELDS:
        sec = config.get(section)
        if not isinstance(sec, dict):
            continue
        if key in sec:
            sec[key] = _resolve_one(sec[key], base)

    for section, key in _LIST_PATH_FIELDS:
        sec = config.get(section)
        if not isinstance(sec, dict):
            continue
        if key in sec and isinstance(sec[key], list):
            sec[key] = [_resolve_one(v, base) for v in sec[key]]

    return config
