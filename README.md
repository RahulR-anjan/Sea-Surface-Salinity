# 🌊 Sea-Surface-Salinity

![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue)
![License: MIT](https://img.shields.io/badge/license-MIT-green)
![Data: OISSS L4](https://img.shields.io/badge/data-OISSS%20L4-informational)

## Overview

This project processes **OISSS L4 Multimission NetCDF** satellite data to study the spatial and temporal variability of sea surface salinity (SSS) in the **Bay of Bengal** (80–100°E, 5–25°N). It includes:

- **Data fetching** from NOAA CoastWatch ERDDAP via `erddapy`
- **NetCDF processing** with a custom HDF5/NetCDF4 reader (no external HDF5 library required)
- **Time series analysis** with linear trend estimation (OLS regression)
- **Anomaly decomposition** and seasonal cycle extraction
- **Smoothing** using the Savitzky-Golay filter
- **Publication-quality dark-themed visualizations**

## Repository Structure

```
Sea-Surface-Salinity/
├── bay_of_bengal_sss_analysis.py  # Core NetCDF processing & analysis
├── bay_of_bengal_sss_plot.py      # Standalone plotting script
├── fetch_smap_salinity.py         # ERDDAP data fetcher
├── data/                          # Place your .nc files here
│   └── .gitkeep
├── requirements.txt
├── .gitignore
├── LICENSE
└── README.md
```

## Setup

### Prerequisites

- Python 3.9+
- NetCDF `.nc` data files (see **Usage** step 1)

### Installation

```bash
git clone https://github.com/RahulR-anjan/Sea-Surface-Salinity.git
cd Sea-Surface-Salinity
pip install -r requirements.txt
```

## Usage

### 1. Fetch SMAP data

```bash
python fetch_smap_salinity.py
```

Downloads OISSS L4 Multimission monthly files from NOAA CoastWatch ERDDAP into the `data/` directory.

### 2. Process NetCDF files

```bash
python bay_of_bengal_sss_analysis.py data/
```

Reads all `.nc` files in the given folder, extracts Bay of Bengal SSS, and writes a CSV time series and an analysis plot.

Optional flags:
```
positional arguments:
  folder        Path to directory containing .nc files
                (default: current dir or $SSS_DATA_DIR)

options:
  -o, --output-dir DIR  Output directory for CSV and plots
                        (default: script directory)
```

### 3. Generate plots

```bash
python bay_of_bengal_sss_plot.py
```

Reads the CSV produced in step 2 and generates additional publication-quality plots.

## Analysis Features

| Feature | Method |
|---|---|
| Linear Trend | OLS regression with p-value |
| Anomaly Decomposition | Deviation from monthly climatology |
| Seasonal Cycle | Monthly mean climatology bar chart |
| Smoothing | Savitzky-Golay (13-month window, order 2) |
| Heatmap | Month × Year SSS heatmap |
| Seasonal Boxplots | DJF / MAM / JJA / SON distributions |

## Study Region

| Parameter | Value |
|---|---|
| Region | Bay of Bengal |
| Longitude | 80–100°E |
| Latitude | 5–25°N |
| Period | Sep 2011 – Dec 2024 |
| Resolution | 0.25° |
| Source | OISSS L4 Multimission |

## Contributing

Contributions are welcome! Areas of interest include:

- Wavelet analysis and EOF decomposition
- Extension to other ocean basins
- Interactive visualizations (Plotly / Bokeh)
- Jupyter notebooks for interactive exploration

Please open an issue or submit a pull request.

## License

This project is licensed under the [MIT License](LICENSE).

## References

- Melnichenko, O., Hacker, P., Potemra, J., & Meissner, T. (2021). *OISSS L4 Multimission Optimally Interpolated Sea Surface Salinity Global Dataset V2.* NASA Physical Oceanography DAAC. https://doi.org/10.5067/SMP20-4U120
- NOAA CoastWatch ERDDAP: https://coastwatch.pfeg.noaa.gov/erddap/