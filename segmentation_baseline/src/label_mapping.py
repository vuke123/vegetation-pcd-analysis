"""
Label mapping utilities for semantic segmentation.

Central place to define class names, IDs, and colour maps so every script
uses the same convention.
"""
from __future__ import annotations

from typing import Dict, List

# Canonical class mapping — matches config/randlanet_config.yaml.
LABEL_MAP: Dict[int, str] = {
    0: "unlabeled",
    1: "vine_row",
    2: "olive_tree",
    3: "other",
}

# Colours for visualization (R, G, B in 0-255).
LABEL_COLOURS: Dict[int, tuple] = {
    0: (128, 128, 128),   # grey
    1: (0, 180, 0),       # green
    2: (139, 90, 43),     # brown
    3: (70, 130, 180),    # steel blue
}

IGNORED_LABEL = 0
NUM_CLASSES = len(LABEL_MAP) - 1  # excludes unlabeled


def id_to_name(label_id: int) -> str:
    return LABEL_MAP.get(label_id, f"unknown_{label_id}")


def name_to_id(name: str) -> int:
    inv = {v: k for k, v in LABEL_MAP.items()}
    if name not in inv:
        raise ValueError(f"Unknown class name: {name!r}. Known: {list(inv)}")
    return inv[name]


def class_names(include_unlabeled: bool = False) -> List[str]:
    """Ordered list of class names."""
    start = 0 if include_unlabeled else 1
    return [LABEL_MAP[i] for i in range(start, len(LABEL_MAP))]
