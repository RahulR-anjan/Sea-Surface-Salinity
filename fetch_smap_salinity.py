"""
fetch_smap_salinity.py
======================
Fetch SMAP sea surface salinity data from NOAA CoastWatch ERDDAP
(dataset: noaacwSMAPsssDaily, griddap, 0.25°, 2015-present),
clean it, and export to CSV.

Requirements:
    pip install erddapy pandas requests

Usage:
    python fetch_smap_salinity.py
"""

import sys
import logging
from pathlib import Path
from datetime import datetime

import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration — edit these as needed
# ---------------------------------------------------------------------------

ERDDAP_SERVER   = "https://coastwatch.pfeg.noaa.gov/erddap"
DATASET_ID      = "noaacwSMAPsssDaily"   # replaces retired jplSMAPSSLv5

# Variables to retrieve — 'sss' is the primary salinity field (PSU)
# 'sss_anom' is the salinity anomaly vs climatology (available in this dataset)
VARIABLES = ["sss", "sss_anom"]

# griddap uses dimension-style constraints: (start):stride:(stop)
# Altitude dimension is always 0.0 for surface products
CONSTRAINTS = {
    "time>=":      "2023-01-01T00:00:00Z",
    "time<=":      "2023-03-31T23:59:59Z",
    "latitude>=":  10.0,
    "latitude<=":  25.0,
    "longitude>=": 80.0,
    "longitude<=": 95.0,
    "altitude>=":  0.0,
    "altitude<=":  0.0,
}

OUTPUT_DIR  = Path(".")
OUTPUT_FILE = OUTPUT_DIR / f"smap_sss_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def build_erddap_client():
    """Initialise and return a configured ERDDAP client."""
    try:
        from erddapy import ERDDAP
    except ImportError:
        log.error("erddapy is not installed. Run: pip install erddapy")
        sys.exit(1)

    log.info("Connecting to ERDDAP server: %s", ERDDAP_SERVER)
    e = ERDDAP(server=ERDDAP_SERVER, protocol="griddap")
    e.dataset_id   = DATASET_ID
    e.constraints  = CONSTRAINTS

    # Only request the columns we care about (skip if VARIABLES is None)
    if VARIABLES:
        e.variables = VARIABLES

    return e


def fetch_data(e) -> pd.DataFrame:
    """Download data and return a pandas DataFrame."""
    log.info("Fetching dataset '%s' …", DATASET_ID)
    log.info("Constraints: %s", CONSTRAINTS)

    try:
        df = e.to_pandas(
            index_col="time (UTC)",
            parse_dates=True,
        )
    except Exception as exc:
        log.error("Failed to fetch data: %s", exc)
        raise

    log.info("Retrieved %d rows and %d columns.", len(df), len(df.columns))
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Basic cleaning: drop all-NaN rows, reset index, rename columns."""
    log.info("Cleaning data …")

    # Drop rows where every value is NaN
    before = len(df)
    df = df.dropna(how="all")
    log.info("Dropped %d all-NaN rows (%d remaining).", before - len(df), len(df))

    # Ensure the time index is timezone-aware UTC
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")

    # Friendly column name aliases
    rename_map = {
        "sss (PSU)":       "sea_surface_salinity_psu",
        "sss_anom (PSU)":  "sea_surface_salinity_anomaly_psu",
        "latitude (degrees_north)": "lat",
        "longitude (degrees_east)": "lon",
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})
    df.index.name = "time_utc"

    return df


def summarise(df: pd.DataFrame) -> None:
    """Print a brief summary of the fetched dataset."""
    print("\n── Dataset summary ─────────────────────────────────")
    print(f"  Rows       : {len(df):,}")
    print(f"  Columns    : {list(df.columns)}")
    print(f"  Time range : {df.index.min()} → {df.index.max()}")
    if "lat" in df.columns:
        print(f"  Lat range  : {df['lat'].min():.3f} → {df['lat'].max():.3f}")
    if "lon" in df.columns:
        print(f"  Lon range  : {df['lon'].min():.3f} → {df['lon'].max():.3f}")
    if "sea_surface_salinity_psu" in df.columns:
        sss = df["sea_surface_salinity_psu"]
        print(f"  SSS (PSU)  : min={sss.min():.3f}, mean={sss.mean():.3f}, max={sss.max():.3f}")
    print("────────────────────────────────────────────────────\n")


def save_csv(df: pd.DataFrame, path: Path) -> None:
    """Write the DataFrame to CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path)
    log.info("Saved output to: %s", path.resolve())


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    e   = build_erddap_client()
    df  = fetch_data(e)
    df  = clean_data(df)
    summarise(df)
    save_csv(df, OUTPUT_FILE)
    log.info("Done.")


if __name__ == "__main__":
    main()