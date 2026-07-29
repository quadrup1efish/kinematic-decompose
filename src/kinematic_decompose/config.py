"""Configuration for kinematic-decompose.

TNG simulation base path and default output directories.
"""
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_BASEPATH_DEFAULT = Path("/Users/yuwa/sims.TNG")

BASEPATH = str(_BASEPATH_DEFAULT)
TEST_IMAGE_PATH = str(_PROJECT_ROOT / "image")
TEST_DATA_PATH = str(_PROJECT_ROOT / "data")
TEST_PATH = str(_PROJECT_ROOT / "tests")
SRC_DIR = str(Path(__file__).resolve().parent.parent)
PROJECT_ROOT = str(_PROJECT_ROOT)

def check_basepath() -> None:
    """Verify that the TNG simulation base path exists. Raises FileNotFoundError if not."""
    if not _BASEPATH_DEFAULT.exists():
        raise FileNotFoundError(
            f"TNG simulation is not in: {_BASEPATH_DEFAULT}\n"
            f"Please set up the correct BASEPATH in config.py"
        )
