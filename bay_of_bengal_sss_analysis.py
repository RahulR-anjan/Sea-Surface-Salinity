"""
Bay of Bengal — Sea Surface Salinity Time Series Analysis
==========================================================
Processes OISSS L4 multimission NetCDF files (Sep 2011 – Dec 2024),
extracts the Bay of Bengal region, computes monthly spatial means,
and produces a full time series analysis with plots.

Requirements:
    pip install netCDF4 numpy pandas matplotlib scipy

Usage:
    1. Set FOLDER_PATH to the directory containing your .nc files.
    2. Run:  python bay_of_bengal_sss_analysis.py
    3. Outputs will be saved in the same folder as this script.
"""

import os
import re
import glob
import zlib
import struct
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.gridspec import GridSpec
from scipy import stats
from scipy.signal import savgol_filter

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION — edit this path
# ─────────────────────────────────────────────────────────────────────────────
FOLDER_PATH = r"C:\path\to\your\netcdf\files"   # <-- change this

# Bay of Bengal bounding box
BOB_LAT_MIN, BOB_LAT_MAX =  5.0, 25.0
BOB_LON_MIN, BOB_LON_MAX = 80.0, 100.0

# Output files
OUTPUT_CSV   = "bay_of_bengal_sss_timeseries.csv"
OUTPUT_PLOT  = "bay_of_bengal_sss_analysis.png"


# ─────────────────────────────────────────────────────────────────────────────
# HDF5/NetCDF4 READER  (no external HDF5 library required)
# ─────────────────────────────────────────────────────────────────────────────

def unshuffle(data: bytes, element_size: int) -> bytes:
    """Reverse the HDF5 shuffle filter."""
    n = len(data) // element_size
    result = bytearray(len(data))
    for i in range(n):
        for j in range(element_size):
            result[i * element_size + j] = data[j * n + i]
    return bytes(result)


def find_tree_nodes(raw: bytes):
    """Find all HDF5 v1 B-tree nodes (TREE signature)."""
    nodes = []
    i = 0
    while True:
        idx = raw.find(b"TREE", i)
        if idx == -1:
            break
        nodes.append(idx)
        i = idx + 1
    return nodes


def read_tree_chunk(raw: bytes, tree_pos: int, key_len: int):
    """
    Read a single-entry chunk B-tree node and return
    (compressed_size, child_file_offset).
    key_len = 24 for 1-D, 32 for 2-D, 40 for 3-D arrays.
    """
    pos = tree_pos + 8 + 16        # skip sig(4)+hdr(4) + two sibling ptrs(8 each)
    compressed_size = struct.unpack_from("<I", raw, pos)[0]
    child = struct.unpack_from("<Q", raw, pos + key_len)[0]
    return compressed_size, child


def decompress_chunk(raw: bytes, child: int, compressed_size: int,
                     element_size: int, dtype: str, shape):
    """Decompress + unshuffle one HDF5 chunk and return numpy array."""
    blob = raw[child: child + compressed_size]
    dec  = zlib.decompress(blob)
    dec  = unshuffle(dec, element_size)
    arr  = np.frombuffer(dec, dtype=dtype)
    if shape:
        arr = arr.reshape(shape)
    return arr


def read_nc_file(filepath: str):
    """
    Minimal NetCDF4/HDF5 reader that extracts:
        longitude (1440,), latitude (720,), sss (720, 1440)
    Returns (lats, lons, sss_2d) as float32 arrays,
    with fill values set to NaN.
    """
    with open(filepath, "rb") as f:
        raw = f.read()

    trees = find_tree_nodes(raw)

    # ── Coordinate arrays (1-D, key_len=24) ──────────────────────────────
    # Longitude: decompresses to 5760 bytes = 1440 × float32
    # Latitude:  decompresses to 2880 bytes = 720  × float32
    coord_trees = [t for t in trees
                   if read_tree_chunk(raw, t, 24)[0] < 5_000]

    lons = lats = None
    for t in trees:
        cs, child = read_tree_chunk(raw, t, 24)
        if cs < 1 or child == 0 or child >= len(raw):
            continue
        try:
            blob = zlib.decompress(raw[child: child + cs])
        except Exception:
            continue
        n_bytes = len(blob)
        if n_bytes == 5760:          # 1440 × 4 bytes  → longitude
            arr = np.frombuffer(unshuffle(blob, 4), dtype="<f4")
            if -181 < arr[0] < 181 and -181 < arr[-1] < 181:
                lons = arr
        elif n_bytes == 2880:        # 720  × 4 bytes  → latitude
            arr = np.frombuffer(unshuffle(blob, 4), dtype="<f4")
            if -91 < arr[0] < 91 and -91 < arr[-1] < 91:
                lats = arr

    # Fall back to computed grids if coordinates not found
    if lons is None:
        lons = np.arange(-179.875, 180.0, 0.25, dtype="float32")
    if lats is None:
        lats = np.arange(-89.875,   90.0, 0.25, dtype="float32")

    # ── SSS array (3-D key_len=40, ~1 MB compressed) ─────────────────────
    sss = None
    for t in trees:
        cs, child = read_tree_chunk(raw, t, 40)
        if cs < 500_000 or child == 0 or child >= len(raw):
            continue
        try:
            arr = decompress_chunk(raw, child, cs,
                                   element_size=4, dtype="<f4",
                                   shape=(720, 1440))
        except Exception:
            continue
        valid = arr[(arr > -500) & (arr < 50)]
        if len(valid) > 100_000 and 20 < valid.mean() < 42:
            sss = arr.astype("float32")
            break

    if sss is None:
        raise ValueError(f"Could not extract SSS from {filepath}")

    # Mask fill values
    sss = sss.astype("float64")
    sss[(sss < -500) | (sss > 50)] = np.nan

    return lats, lons, sss


# ─────────────────────────────────────────────────────────────────────────────
# PARSE DATE FROM FILENAME
# ─────────────────────────────────────────────────────────────────────────────

def parse_date(filepath: str):
    """Extract YYYY-MM from filename."""
    name = os.path.basename(filepath)
    # Matches patterns like 2022-07, 202207, v1_0_2022-07, etc.
    m = re.search(r"(\d{4})[-_]?(\d{2})", name)
    if m:
        return int(m.group(1)), int(m.group(2))
    raise ValueError(f"Cannot parse date from filename: {name}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN PROCESSING LOOP
# ─────────────────────────────────────────────────────────────────────────────

def process_files(folder: str):
    files = sorted(glob.glob(os.path.join(folder, "*.nc")))
    if not files:
        raise FileNotFoundError(f"No .nc files found in: {folder}")

    print(f"Found {len(files)} NetCDF files. Processing...\n")

    records = []
    for i, fp in enumerate(files, 1):
        try:
            year, month = parse_date(fp)
            lats, lons, sss = read_nc_file(fp)

            # Slice to Bay of Bengal
            lat_mask = (lats >= BOB_LAT_MIN) & (lats <= BOB_LAT_MAX)
            lon_mask = (lons >= BOB_LON_MIN) & (lons <= BOB_LON_MAX)
            bob_sss  = sss[np.ix_(lat_mask, lon_mask)]

            mean_sss  = float(np.nanmean(bob_sss))
            std_sss   = float(np.nanstd(bob_sss))
            n_valid   = int(np.sum(~np.isnan(bob_sss)))

            records.append({
                "year":      year,
                "month":     month,
                "date":      pd.Timestamp(year=year, month=month, day=15),
                "sss_mean":  round(mean_sss, 4),
                "sss_std":   round(std_sss, 4),
                "n_valid":   n_valid,
            })
            print(f"  [{i:3d}/{len(files)}] {year}-{month:02d}  SSS mean = {mean_sss:.3f} psu  "
                  f"(n={n_valid:,})")

        except Exception as e:
            print(f"  [{i:3d}/{len(files)}] SKIPPED {os.path.basename(fp)}: {e}")

    df = pd.DataFrame(records).sort_values("date").reset_index(drop=True)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# ANALYSIS & PLOTTING
# ─────────────────────────────────────────────────────────────────────────────

def analyse_and_plot(df: pd.DataFrame, out_csv: str, out_plot: str):

    # ── Save CSV ──────────────────────────────────────────────────────────
    df.to_csv(out_csv, index=False)
    print(f"\nTime series saved → {out_csv}")

    # ── Seasonal climatology ──────────────────────────────────────────────
    clim = df.groupby("month")["sss_mean"].mean()
    df["sss_anomaly"] = df.apply(
        lambda r: r["sss_mean"] - clim[r["month"]], axis=1
    )

    # ── Linear trend ─────────────────────────────────────────────────────
    x = np.arange(len(df))
    slope, intercept, r, p, se = stats.linregress(x, df["sss_mean"])
    trend_line = slope * x + intercept
    trend_per_decade = slope * 12 * 10   # months → decade

    # ── Smoothed series ───────────────────────────────────────────────────
    if len(df) >= 13:
        smooth = savgol_filter(df["sss_mean"], window_length=13, polyorder=2)
    else:
        smooth = df["sss_mean"].values

    # ── Seasonal means ────────────────────────────────────────────────────
    season_map = {12: "DJF", 1: "DJF", 2: "DJF",
                  3:  "MAM", 4: "MAM", 5: "MAM",
                  6:  "JJA", 7: "JJA", 8: "JJA",
                  9:  "SON", 10: "SON", 11: "SON"}
    df["season"] = df["month"].map(season_map)
    season_means = df.groupby("season")["sss_mean"].mean().reindex(
        ["DJF", "MAM", "JJA", "SON"]
    )

    # ── PLOT ─────────────────────────────────────────────────────────────
    plt.rcParams.update({
        "font.family":     "DejaVu Sans",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid":       True,
        "grid.alpha":      0.3,
        "grid.linestyle":  "--",
    })

    fig = plt.figure(figsize=(16, 14), facecolor="#0d1117")
    fig.suptitle(
        "Bay of Bengal  ·  Sea Surface Salinity  ·  Sep 2011 – Dec 2024",
        fontsize=16, fontweight="bold", color="white", y=0.98
    )

    gs = GridSpec(3, 2, figure=fig, hspace=0.45, wspace=0.35,
                  top=0.93, bottom=0.07, left=0.08, right=0.96)

    OCEAN  = "#0d1117"
    C1     = "#00d4ff"   # cyan
    C2     = "#ff6b6b"   # coral
    C3     = "#ffd166"   # yellow
    C4     = "#06d6a0"   # mint
    TXT    = "#e0e0e0"
    GRID   = "#1e2a38"

    def style_ax(ax, title):
        ax.set_facecolor(OCEAN)
        ax.tick_params(colors=TXT, labelsize=9)
        ax.xaxis.label.set_color(TXT)
        ax.yaxis.label.set_color(TXT)
        ax.title.set_color(TXT)
        ax.set_title(title, fontsize=11, fontweight="bold", pad=8)
        for spine in ax.spines.values():
            spine.set_color(GRID)
        ax.grid(color=GRID, linestyle="--", alpha=0.6)

    dates = df["date"]

    # ── Panel 1: Full time series ─────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, :])
    ax1.fill_between(dates, df["sss_mean"], alpha=0.15, color=C1)
    ax1.plot(dates, df["sss_mean"], color=C1, lw=1.2,
             alpha=0.7, label="Monthly mean SSS")
    ax1.plot(dates, smooth, color=C3, lw=2.0, label="13-month smooth")
    ax1.plot(dates, trend_line, color=C2, lw=1.8, ls="--",
             label=f"Trend  {trend_per_decade:+.3f} psu/decade  (p={p:.3f})")
    ax1.set_ylabel("SSS (psu)", color=TXT)
    ax1.legend(fontsize=9, facecolor="#1a2332", labelcolor=TXT,
               edgecolor=GRID, loc="upper right")
    style_ax(ax1, "Monthly Mean Sea Surface Salinity — Bay of Bengal")
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax1.xaxis.set_major_locator(mdates.YearLocator(2))

    # ── Panel 2: Anomaly ──────────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[1, :])
    pos = df["sss_anomaly"] >= 0
    ax2.bar(dates[pos],  df["sss_anomaly"][pos],  color=C2,
            width=25, label="Positive anomaly")
    ax2.bar(dates[~pos], df["sss_anomaly"][~pos], color=C4,
            width=25, label="Negative anomaly")
    ax2.axhline(0, color=TXT, lw=0.8, ls="-")
    ax2.set_ylabel("SSS Anomaly (psu)", color=TXT)
    ax2.legend(fontsize=9, facecolor="#1a2332", labelcolor=TXT,
               edgecolor=GRID)
    style_ax(ax2, "SSS Anomaly (deviation from monthly climatology)")
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax2.xaxis.set_major_locator(mdates.YearLocator(2))

    # ── Panel 3: Seasonal cycle ───────────────────────────────────────────
    ax3 = fig.add_subplot(gs[2, 0])
    month_labels = ["Jan","Feb","Mar","Apr","May","Jun",
                    "Jul","Aug","Sep","Oct","Nov","Dec"]
    clim_vals = [clim.get(m, np.nan) for m in range(1, 13)]
    bars = ax3.bar(month_labels, clim_vals, color=C1, alpha=0.8, edgecolor=GRID)
    ax3.set_ylabel("Mean SSS (psu)", color=TXT)
    ax3.set_xlabel("Month", color=TXT)
    # Annotate bars
    for bar, v in zip(bars, clim_vals):
        if not np.isnan(v):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                     f"{v:.2f}", ha="center", va="bottom",
                     fontsize=7, color=TXT)
    style_ax(ax3, "Mean Seasonal Cycle")

    # ── Panel 4: Seasonal box / summary stats ────────────────────────────
    ax4 = fig.add_subplot(gs[2, 1])
    seasons    = ["DJF", "MAM", "JJA", "SON"]
    season_col = {"DJF": C3, "MAM": C4, "JJA": C2, "SON": C1}
    data_by_season = [df[df["season"] == s]["sss_mean"].values for s in seasons]
    bp = ax4.boxplot(data_by_season, labels=seasons, patch_artist=True,
                     medianprops=dict(color="white", lw=2),
                     whiskerprops=dict(color=TXT),
                     capprops=dict(color=TXT),
                     flierprops=dict(markerfacecolor=TXT, markersize=4))
    for patch, s in zip(bp["boxes"], seasons):
        patch.set_facecolor(season_col[s])
        patch.set_alpha(0.75)
    ax4.set_ylabel("SSS (psu)", color=TXT)
    ax4.set_xlabel("Season", color=TXT)
    style_ax(ax4, "SSS Distribution by Season")

    # ── Summary stats text box ────────────────────────────────────────────
    overall_mean = df["sss_mean"].mean()
    overall_std  = df["sss_mean"].std()
    overall_min  = df["sss_mean"].min()
    overall_max  = df["sss_mean"].max()
    min_row = df.loc[df["sss_mean"].idxmin()]
    max_row = df.loc[df["sss_mean"].idxmax()]

    stats_text = (
        f"Overall mean:   {overall_mean:.3f} psu\n"
        f"Std deviation:  {overall_std:.3f} psu\n"
        f"Min:  {overall_min:.3f}  ({min_row['year']}-{min_row['month']:02d})\n"
        f"Max:  {overall_max:.3f}  ({max_row['year']}-{max_row['month']:02d})\n"
        f"Trend:  {trend_per_decade:+.4f} psu / decade\n"
        f"R²:  {r**2:.3f}   p-value:  {p:.4f}"
    )
    fig.text(0.5, 0.005, stats_text,
             ha="center", va="bottom", fontsize=9,
             color=TXT, fontfamily="monospace",
             bbox=dict(facecolor="#1a2332", edgecolor=GRID,
                       boxstyle="round,pad=0.5", alpha=0.8))

    plt.savefig(out_plot, dpi=150, bbox_inches="tight", facecolor=OCEAN)
    plt.close()
    print(f"Plot saved        → {out_plot}")

    # ── Print summary ────────────────────────────────────────────────────
    print("\n" + "="*55)
    print("  BAY OF BENGAL SSS — SUMMARY STATISTICS")
    print("="*55)
    print(f"  Period:          {df['date'].min().strftime('%b %Y')} – "
          f"{df['date'].max().strftime('%b %Y')}")
    print(f"  Months:          {len(df)}")
    print(f"  Overall mean:    {overall_mean:.3f} psu")
    print(f"  Std deviation:   {overall_std:.3f} psu")
    print(f"  Min SSS:         {overall_min:.3f}  ({min_row['year']}-{min_row['month']:02d})")
    print(f"  Max SSS:         {overall_max:.3f}  ({max_row['year']}-{max_row['month']:02d})")
    print(f"  Trend:           {trend_per_decade:+.4f} psu / decade")
    print(f"  R²:              {r**2:.3f}")
    print(f"  p-value:         {p:.4f}  "
          f"({'significant' if p < 0.05 else 'not significant'} at 5%)")
    print("="*55)
    print("\n  Seasonal means:")
    for s in ["DJF", "MAM", "JJA", "SON"]:
        print(f"    {s}:  {season_means[s]:.3f} psu")
    print()


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if not os.path.isdir(FOLDER_PATH):
        print(f"ERROR: Folder not found: {FOLDER_PATH}")
        print("Please edit FOLDER_PATH at the top of this script.")
        exit(1)

    df = process_files(FOLDER_PATH)

    if df.empty:
        print("No data extracted. Check your file paths and formats.")
        exit(1)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    out_csv  = os.path.join(script_dir, OUTPUT_CSV)
    out_plot = os.path.join(script_dir, OUTPUT_PLOT)

    analyse_and_plot(df, out_csv, out_plot)
