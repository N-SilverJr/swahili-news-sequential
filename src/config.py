"""Central configuration shared by every notebook.

All five approaches import their settings from here, so they use the SAME split, the SAME text
cleaning and the SAME evaluation. Change values here (not inside notebooks) and record the change
in the experiment log.
"""
import os
from pathlib import Path

# ---------------------------------------------------------------- paths
ROOT = Path(os.environ.get("PROJECT_ROOT", Path(__file__).resolve().parents[1]))
DATA_DIR = Path(os.environ.get("DATA_DIR", ROOT / "data"))   # Train.csv, Test.csv, SampleSubmission.csv
CACHE_DIR = Path(os.environ.get("CACHE_DIR", DATA_DIR / "cache"))
SPLITS_PATH = ROOT / "data" / "splits.csv"                    # committed so all members share it
RESULTS_DIR = ROOT / "results"
FIG_DIR = RESULTS_DIR / "figures"
EXP_LOG = ROOT / "experiments" / "experiment_log.csv"
SUBMISSION_DIR = ROOT / "submissions"

# ---------------------------------------------------------------- data
TEXT_COL, LABEL_COL, ID_COL = "content", "category", "id"
TEST_ID_COL = "swahili_id"
# Order of the probability columns in Zindi's SampleSubmission.csv
CLASSES = ["kitaifa", "michezo", "biashara", "kimataifa", "burudani"]
ENGLISH = {"kitaifa": "national", "michezo": "sports", "biashara": "business",
           "kimataifa": "international", "burudani": "entertainment"}

# ---------------------------------------------------------------- split
SEED = 42
VAL_SIZE = 0.15     # 15% / 15% instead of 10% / 10% so the tiny classes (17 and 54 articles)
TEST_SIZE = 0.15    # still have a few examples in validation and test

# ---------------------------------------------------------------- sequence settings
MAX_WORDS = int(os.environ.get("MAX_WORDS", 512))   # word-level models (CNN / RNN); ~90th percentile, see EDA
MIN_FREQ = 2                                        # vocabulary: keep words seen at least twice in TRAIN
MAX_VOCAB = 40_000
MAX_TOKENS = 512                                    # transformer sub-word limit (architectural)

QUICK = os.environ.get("QUICK") == "1"   # smoke-test mode: tiny epochs, only to check the code runs
if QUICK:   # never mix smoke-test outputs with real results
    RESULTS_DIR, EXP_LOG = ROOT / "_quick" / "results", ROOT / "_quick" / "experiments" / "experiment_log.csv"
    FIG_DIR, SUBMISSION_DIR = RESULTS_DIR / "figures", ROOT / "_quick" / "submissions"

for _d in (CACHE_DIR, RESULTS_DIR, FIG_DIR, EXP_LOG.parent, SUBMISSION_DIR, SPLITS_PATH.parent):
    _d.mkdir(parents=True, exist_ok=True)
