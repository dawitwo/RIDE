from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
MODELS = ROOT / "models"
FIGURES = ROOT / "figures"
SUBMISSION = ROOT / "submission"
APP_ASSETS = ROOT / "app" / "assets"

RAW_FILES = {
    "train": DATA_RAW / "ride_demand_train.csv",
    "test": DATA_RAW / "ride_demand_test.csv",
    "submission_template": DATA_RAW / "submission_template.csv",
    "weather": DATA_RAW / "weather_hourly.csv",
    "events": DATA_RAW / "events_calendar.csv",
}

TIMEZONE = "Africa/Addis_Ababa"
RANDOM_STATE = 42

TRAIN_START = "2025-01-01"
TRAIN_END = "2025-10-31"
VALID_START = "2025-10-18"
VALID_END = "2025-10-31"
TEST_START = "2025-11-01"
TEST_END = "2025-11-14"

# Fill after inspecting raw unique values. Empty is fine to start.
CANONICAL_ZONES = []
ZONE_MAP = {}
EVENT_TYPE_MAP = {}
