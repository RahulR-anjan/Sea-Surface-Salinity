# Sea Surface Salinity — Bay of Bengal

Analysis of sea surface salinity (SSS) variability in the Bay of Bengal using NASA SMAP satellite data.

<!-- One line here on WHY this matters, e.g.:
The Bay of Bengal has one of the most complex salinity regimes in the world ocean,
driven by heavy monsoon river discharge (Ganges-Brahmaputra) and seasonal freshwater
plumes. Understanding SSS variability here is relevant to monsoon dynamics, cyclone
intensification, and coastal/estuarine ecosystems including the Sundarbans. -->

## What this does

- **`fetch_smap_salinity.py`** — Downloads/retrieves SMAP (Soil Moisture Active Passive) sea surface salinity data for the Bay of Bengal region
  <!-- add: data source/API, spatial extent (lat/lon bounds), time range covered -->
- **`bay_of_bengal_sss_analysis.py`** — [describe: what analysis is performed — e.g. seasonal averaging, trend detection, anomaly calculation, correlation with river discharge/rainfall]
- **`bay_of_bengal_sss_plot.py`** — Generates visualizations of SSS patterns
  <!-- add: what kind of plots — time series, spatial maps, seasonal comparison? -->

## Data

- **Source:** NASA SMAP (Soil Moisture Active Passive) satellite
- **Region:** Bay of Bengal
- **Time period:** <!-- fill in date range -->
- **Resolution:** <!-- spatial/temporal resolution if known -->

## Key findings

<!-- This is the most important section for a reader — even 2-3 bullet points
matter far more than the code itself. Examples of what to fill in:
- Seasonal salinity minimum observed during [month(s)], consistent with monsoon
  freshwater discharge from the Ganges-Brahmaputra system
- SSS ranged from X to Y PSU across the study period
- [Any spatial pattern you found — e.g. stronger freshening near the coast/river mouth
  vs. open bay]
-->

![Sample plot](path/to/your/plot.png)
<!-- Embed your best plot here — this single image does more for the repo than
any amount of text. Add it to the repo (e.g. in an /outputs or /figures folder)
and update the path. -->

## How to run

```bash
# Clone and set up
git clone https://github.com/RahulR-anjan/Sea-Surface-Salinity.git
cd Sea-Surface-Salinity
pip install -r requirements.txt   # create this if it doesn't exist yet

# Run the pipeline
python fetch_smap_salinity.py
python bay_of_bengal_sss_analysis.py
python bay_of_bengal_sss_plot.py
```

**Dependencies:** <!-- list key packages, e.g. numpy, pandas, xarray, matplotlib, netCDF4 -->

## Project structure

```
Sea-Surface-Salinity/
├── README.md
├── fetch_smap_salinity.py       # Data retrieval
├── bay_of_bengal_sss_analysis.py # Core analysis
├── bay_of_bengal_sss_plot.py     # Visualization
└── requirements.txt               # (add this)
```

## Motivation / context

<!-- Optional but valuable for a PhD-facing repo: 1-2 sentences on why you chose
this — course project? personal interest tied to your earth science background?
connection to your Sundarbans article/coastal vulnerability interests? Reviewers
like seeing a thread connecting your work. -->

## Future work

<!-- Optional: e.g. extend to multi-year trend analysis, correlate with cyclone
data, compare with in-situ buoy measurements, ML-based prediction -->

---
*Part of ongoing work in applied ML/environmental science. See also: [air-quality-dashboard](link).*
