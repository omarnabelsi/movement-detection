"""
YAML configuration loader with sensible defaults.

The configuration file is a flat, human-readable YAML document.
Missing keys fall back to the built-in DEFAULT_CONFIG.
"""
import yaml
from pathlib import Path

DEFAULT_CONFIG: dict = {
    "source": {
        "type": "screen",        # camera | screen | video
        "camera_index": 0,
        "monitor_index": 1,
        "video_path": "",
    },
    "detection": {
        "history": 500,          # MOG2 background model length
        "var_threshold": 25,     # MOG2 sensitivity (lower = more sensitive)
        "min_area": 500,         # Minimum contour area in pixels
        "blur_size": 21,         # Gaussian blur kernel size (odd number)
        "erode_iterations": 1,   # Noise removal passes
        "dilate_iterations": 3,  # Gap-filling passes
    },
    "tracker": {
        "max_disappeared": 30,   # Frames before dropping a lost object
    },
    "performance": {
        "target_fps": 30,        # Maximum frames per second
        "resize_width": 640,     # Frame width for processing (0 = no resize)
    },
}


def load_config(path: str = "config.yaml") -> dict:
    """Load configuration from a YAML file.

    Falls back to DEFAULT_CONFIG if the file is missing or invalid.
    Individual missing keys are filled with defaults via deep merge.
    """
    p = Path(path)
    if p.exists():
        try:
            with open(p, encoding="utf-8") as f:
                user = yaml.safe_load(f)
            if isinstance(user, dict):
                return _deep_merge(DEFAULT_CONFIG, user)
        except Exception:
            pass
    return _copy(DEFAULT_CONFIG)


def save_config(config: dict, path: str = "config.yaml"):
    """Persist configuration to a YAML file."""
    try:
        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    except Exception as e:
        print(f"[config] Error saving: {e}")


# ── Helpers ──────────────────────────────────────────────────────────

def _deep_merge(defaults: dict, overrides: dict) -> dict:
    """Recursively merge *overrides* into a copy of *defaults*."""
    merged = dict(defaults)
    for key, val in overrides.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(val, dict):
            merged[key] = _deep_merge(merged[key], val)
        else:
            merged[key] = val
    return merged


def _copy(d: dict) -> dict:
    import copy
    return copy.deepcopy(d)
