from pathlib import Path
BASEPATH = Path("/Users/yuwa/sims.TNG")
if not BASEPATH.exists():
    raise FileNotFoundError(
        f"TNG simulation is not in: {BASEPATH}\n"
        f"Please set up the correct BASEPATH in config.py"
    )
src_dir = Path(__file__).resolve().parent.parent
TEST_IMAGE_PATH = src_dir.parent/"image"
TEST_DATA_PATH  = src_dir.parent/"data"
TEST_PATH  = src_dir.parent/"tests"

BASEPATH = str(BASEPATH)
TEST_IMAGE_PATH = str(TEST_IMAGE_PATH)
TEST_DATA_PATH = str(TEST_DATA_PATH)
TEST_PATH = str(TEST_PATH)
SRC_DIR = str(src_dir)
PROJECT_ROOT = str(src_dir.parent)
