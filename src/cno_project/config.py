"""Configuration for the CNO forecasting pipeline."""

from pathlib import Path

BASE_DIR = Path.cwd()

DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
FINAL_DIR = DATA_DIR / "final"

OUTPUT_DIR = BASE_DIR / "outputs"
TABLES_DIR = OUTPUT_DIR / "tables"
BACKTEST_DIR = OUTPUT_DIR / "backtests"

for directory in [
    RAW_DIR,
    PROCESSED_DIR,
    FINAL_DIR,
    OUTPUT_DIR,
    TABLES_DIR,
    BACKTEST_DIR,
]:
    directory.mkdir(parents=True, exist_ok=True)

START_DATE = "2010-05-01"

SEED = 42
TRAIN_SIZE = 48
N_LAGS = 4
KAPPA = 0.63

ONI_URL = "https://www.cpc.ncep.noaa.gov/data/indices/oni.ascii.txt"

RAW_FILES = {
    "cno": "cno_export_prices.csv",
    "production": "coconut_production.csv",
    "copra": "copra_farmgate.csv",
    "cpo": "cpo_futures.csv",
    "oni": "oni_monthly.csv",
    "usd_php": "usd_php.csv",
}