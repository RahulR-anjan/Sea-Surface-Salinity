"""
Bay of Bengal — Sea Surface Salinity  |  Python Plotting Script
================================================================
Reads bay_of_bengal_sss_timeseries.csv and produces a 4-panel figure:
  1. Full time series with trend & smooth
  2. Monthly anomaly bars
  3. Mean seasonal cycle with error bars
  4. Year × Month heatmap

Requirements:
    pip install pandas numpy matplotlib scipy

Usage:
    python bay_of_bengal_sss_plot.py
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')            # headless rendering (safe for all systems)
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.dates as mdates
from matplotlib.colors import LinearSegmentedColormap
from scipy.signal import savgol_filter
from scipy import stats

# ── Configuration ────────────────────────────────────────────────────────────
CSV_FILE  = 'bay_of_bengal_sss_timeseries.csv'   # <-- path to your CSV
OUT_FILE  = 'bay_of_bengal_sss_analysis.png'
DPI       = 160

# ── Colour palette ────────────────────────────────────────────────────────────
BG      = '#050d14'
SURFACE = '#091724'
BORDER  = '#0e2336'
CYAN    = '#00c8e0'
AMBER   = '#e0a020'
CORAL   = '#e05050'
GREEN   = '#5ee8b0'
MUTED   = '#4a6e88'
WHITE   = '#eaf4fb'
TEXT    = '#c8dce8'

# ── Load & enrich data ────────────────────────────────────────────────────────
df = pd.read_csv(CSV_FILE, parse_dates=['date'])
df = df.sort_values('date').reset_index(drop=True)

# Monthly climatology & anomaly
clim = df.groupby('month')['sss_mean'].mean()
df['anomaly'] = df.apply(lambda r: r['sss_mean'] - clim[r['month']], axis=1)

# Linear trend
x_idx = np.arange(len(df))
slope, intercept, r_val, p_val, _ = stats.linregress(x_idx, df['sss_mean'])
df['trend']         = slope * x_idx + intercept
trend_per_decade    = slope * 12 * 10

# 12-month smooth (Savitzky-Golay)
df['smooth'] = savgol_filter(df['sss_mean'], window_length=13, polyorder=2)

# Seasonal climatology arrays
month_names  = ['Jan','Feb','Mar','Apr','May','Jun',
                'Jul','Aug','Sep','Oct','Nov','Dec']
season_clim  = [clim[m] for m in range(1, 13)]
season_std   = [df[df['month'] == m]['sss_std'].mean() for m in range(1, 13)]

# Heatmap pivot (years × months)
years = sorted(df['year'].unique())
hmap  = np.full((len(years), 12), np.nan)
for i, yr in enumerate(years):
    for j, mo in enumerate(range(1, 13)):
        row = df[(df['year'] == yr) & (df['month'] == mo)]
        if not row.empty:
            hmap[i, j] = row['sss_mean'].values[0]

# Summary stats
overall_mean = df['sss_mean'].mean()
min_row = df.loc[df['sss_mean'].idxmin()]
max_row = df.loc[df['sss_mean'].idxmax()]

# ── Global style ──────────────────────────────────────────────────────────────
plt.rcParams.update({
    'figure.facecolor':  BG,
    'axes.facecolor':    SURFACE,
    'axes.edgecolor':    BORDER,
    'axes.labelcolor':   MUTED,
    'axes.titlecolor':   WHITE,
    'xtick.color':       MUTED,
    'ytick.color':       MUTED,
    'xtick.labelsize':   8,
    'ytick.labelsize':   8,
    'grid.color':        BORDER,
    'grid.linestyle':    '--',
    'grid.linewidth':    0.6,
    'text.color':        TEXT,
    'font.family':       'monospace',
    'axes.spines.top':   False,
    'axes.spines.right': False,
})

# ── Figure layout ─────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(18, 14))
fig.patch.set_facecolor(BG)

gs = gridspec.GridSpec(
    3, 2, figure=fig,
    height_ratios=[2.2, 1.4, 2],
    hspace=0.52, wspace=0.28,
    left=0.07, right=0.97, top=0.90, bottom=0.07,
)
ax_ts   = fig.add_subplot(gs[0, :])
ax_anom = fig.add_subplot(gs[1, :])
ax_sea  = fig.add_subplot(gs[2, 0])
ax_hm   = fig.add_subplot(gs[2, 1])

# ── Page header ───────────────────────────────────────────────────────────────
fig.text(0.07, 0.955, 'BAY OF BENGAL  ·  SEA SURFACE SALINITY',
         fontsize=11, color=CYAN, fontweight='bold', fontfamily='monospace')
fig.text(0.07, 0.928, 'Sea Surface Salinity',
         fontsize=26, color=WHITE, fontstyle='italic')
fig.text(0.07, 0.902,
         'Monthly spatial means  ·  Sep 2011 – Dec 2024  ·  OISSS L4 Multimission'
         '  ·  80–100°E, 5–25°N',
         fontsize=8.5, color=MUTED, fontfamily='monospace')

stats_str = (
    f"Mean {overall_mean:.3f} psu   |   "
    f"Min {min_row['sss_mean']:.3f} ({int(min_row['year'])}-{int(min_row['month']):02d})   |   "
    f"Max {max_row['sss_mean']:.3f} ({int(max_row['year'])}-{int(max_row['month']):02d})   |   "
    f"Trend {trend_per_decade:+.3f} psu/decade   |   p = {p_val:.4f}"
)
fig.text(0.97, 0.958, stats_str, fontsize=8, color=CYAN,
         ha='right', fontfamily='monospace')


# ════════════════════════════════════════════════════════════════════════════
# Panel 1 — Full Time Series
# ════════════════════════════════════════════════════════════════════════════
ax = ax_ts
ax.grid(True, axis='y', alpha=0.4)

# ±0.5σ shaded envelope
ax.fill_between(df['date'],
                df['sss_mean'] - df['sss_std'] * 0.5,
                df['sss_mean'] + df['sss_std'] * 0.5,
                alpha=0.10, color=CYAN)

# 2024 freshening highlight
ax.axvspan(pd.Timestamp('2024-01-01'), df['date'].max(),
           color=CORAL, alpha=0.06, zorder=0)
ax.text(pd.Timestamp('2024-02-01'),
        df['sss_mean'].max() + 0.04,
        '2024 freshening', fontsize=7.5, color=CORAL,
        fontfamily='monospace', alpha=0.8)

# Trend line
ax.plot(df['date'], df['trend'], color=CORAL, lw=1.5, ls='--', alpha=0.8, zorder=2,
        label=f'Trend  {trend_per_decade:+.3f} psu/decade  (p = {p_val:.3f})')

# 12-month smooth
ax.plot(df['date'], df['smooth'], color=AMBER, lw=2.2, zorder=3,
        label='12-month smooth')

# Raw monthly line
ax.plot(df['date'], df['sss_mean'], color=CYAN, lw=1.0, alpha=0.75, zorder=4,
        label='Monthly mean SSS')

ax.set_title('Time Series', fontsize=10, color=WHITE,
             fontfamily='monospace', loc='left', pad=8)
ax.set_ylabel('SSS (psu)', fontsize=9)
ax.xaxis.set_major_locator(mdates.YearLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
ax.set_xlim(df['date'].min(), df['date'].max())
ax.legend(fontsize=8, framealpha=0, labelcolor=TEXT, loc='lower left', ncol=4)

# Annotate min/max
for row, label, yoff in [
    (min_row, f"min\n{min_row['sss_mean']:.2f}", -22),
    (max_row, f"max\n{max_row['sss_mean']:.2f}",  16),
]:
    ax.annotate(label,
                xy=(row['date'], row['sss_mean']),
                xytext=(0, yoff), textcoords='offset points',
                fontsize=7, color=MUTED, ha='center', fontfamily='monospace',
                arrowprops=dict(arrowstyle='->', color=MUTED, lw=0.8))

ax.spines['left'].set_color(BORDER)
ax.spines['bottom'].set_color(BORDER)


# ════════════════════════════════════════════════════════════════════════════
# Panel 2 — Anomaly Bars
# ════════════════════════════════════════════════════════════════════════════
ax = ax_anom
ax.grid(True, axis='y', alpha=0.4)
ax.axhline(0, color=TEXT, lw=0.8, alpha=0.3)

pos = df['anomaly'] >= 0
ax.bar(df.loc[pos,  'date'], df.loc[pos,  'anomaly'], width=20,
       color=CORAL, alpha=0.85, label='Saltier than normal')
ax.bar(df.loc[~pos, 'date'], df.loc[~pos, 'anomaly'], width=20,
       color=GREEN, alpha=0.85, label='Fresher than normal')

ax.set_title('Anomaly  (deviation from monthly climatology)',
             fontsize=10, color=WHITE, fontfamily='monospace', loc='left', pad=8)
ax.set_ylabel('SSS Anomaly (psu)', fontsize=9)
ax.xaxis.set_major_locator(mdates.YearLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
ax.set_xlim(df['date'].min(), df['date'].max())
ax.legend(fontsize=8, framealpha=0, labelcolor=TEXT, loc='lower left', ncol=2)
ax.spines['left'].set_color(BORDER)
ax.spines['bottom'].set_color(BORDER)


# ════════════════════════════════════════════════════════════════════════════
# Panel 3 — Mean Seasonal Cycle
# ════════════════════════════════════════════════════════════════════════════
ax = ax_sea
ax.grid(True, axis='y', alpha=0.4)

xpos = np.arange(12)
ax.bar(xpos, season_clim, color=CYAN, alpha=0.22,
       edgecolor=CYAN, linewidth=0.9, width=0.65)
ax.errorbar(xpos, season_clim, yerr=season_std, fmt='none',
            ecolor=AMBER, elinewidth=1.5, capsize=4, capthick=1.5, zorder=5)

for i, (v, s) in enumerate(zip(season_clim, season_std)):
    ax.text(i, v + s + 0.05, f'{v:.2f}',
            ha='center', va='bottom', fontsize=7.5,
            color=TEXT, fontfamily='monospace')

ax.set_xticks(xpos)
ax.set_xticklabels(month_names, fontsize=8)
ax.set_title('Mean Seasonal Cycle', fontsize=10, color=WHITE,
             fontfamily='monospace', loc='left', pad=8)
ax.set_ylabel('Mean SSS (psu)', fontsize=9)
ax.set_ylim(min(season_clim) - max(season_std) - 0.5,
            max(season_clim) + max(season_std) + 0.6)
ax.spines['left'].set_color(BORDER)
ax.spines['bottom'].set_color(BORDER)


# ════════════════════════════════════════════════════════════════════════════
# Panel 4 — Year × Month Heatmap
# ════════════════════════════════════════════════════════════════════════════
ax = ax_hm

cmap = LinearSegmentedColormap.from_list(
    'ocean_sss',
    ['#083d6e', '#0e6ea8', '#00c8e0', '#a8eaf5', '#fdf6e3', '#f5c27a', '#e07020'],
    N=256,
)
vmin, vmax = np.nanmin(hmap), np.nanmax(hmap)
im = ax.imshow(hmap, aspect='auto', cmap=cmap,
               vmin=vmin, vmax=vmax, interpolation='nearest')

# Cell value labels
for i in range(len(years)):
    for j in range(12):
        if not np.isnan(hmap[i, j]):
            bright = (hmap[i, j] - vmin) / (vmax - vmin)
            tc = '#05101a' if bright > 0.55 else WHITE
            ax.text(j, i, f'{hmap[i, j]:.1f}',
                    ha='center', va='center',
                    fontsize=5.8, color=tc,
                    fontfamily='monospace', fontweight='bold')

ax.set_xticks(range(12))
ax.set_xticklabels([m[0] for m in month_names], fontsize=8)
ax.set_yticks(range(len(years)))
ax.set_yticklabels(years, fontsize=7)
ax.set_title('Year × Month Heatmap', fontsize=10, color=WHITE,
             fontfamily='monospace', loc='left', pad=8)

cbar = plt.colorbar(im, ax=ax, orientation='horizontal',
                    pad=0.06, fraction=0.04, aspect=30)
cbar.ax.tick_params(labelsize=7, colors=MUTED)
cbar.set_label('SSS (psu)', fontsize=8, color=MUTED)
cbar.outline.set_edgecolor(BORDER)
for spine in ax.spines.values():
    spine.set_edgecolor(BORDER)

# ── Save ─────────────────────────────────────────────────────────────────────
plt.savefig(OUT_FILE, dpi=DPI, bbox_inches='tight', facecolor=BG)
plt.close()
print(f"✓  Saved → {OUT_FILE}")
