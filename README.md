# 🌲🔥 Forest Fire Mapping — India

A multi-step pipeline for mapping and analyzing forest fires across India using
MODIS fire detections and ESA-CCI/C3S land-use/land-cover (LULC) data.

> **This repository currently covers Step 1: Forest Fire Points Extraction.**
> Later steps (NDVI-based vegetation stress, land-surface-temperature analysis,
> ML-based risk modeling) build on the outputs produced here and will be added
> as separate steps.

---

## Step 1 — Forest Fire Points Extraction

**Notebook:** [`FOREST_FIRE_POINTS_EXTRACTION(INDIA).ipynb`](FOREST_FIRE_POINTS_EXTRACTION(INDIA).ipynb)

**Reference methodology:** Biswas et al. (2025) & Uthappa et al. (2025), forest
classes per Sannigrahi et al. (2018)

**Study period:** 1 Nov 2000 – 15 Dec 2022 (capped at 2022 — the last year with
published ESA-CCI/C3S LULC data)

### What it does

1. Loads the MODIS Collection 6.1 fire archive (FIRMS) for India.
2. Clips fire detections to India's *exact* polygon (not just a lon/lat
   bounding box — a bbox also covers Sri Lanka, Nepal, Bangladesh, Myanmar,
   and southern Pakistan, so an exact point-in-polygon test is used instead).
3. Extracts the yearly ESA-CCI/C3S LULC ZIP archives and builds a binary
   forest mask per year from the Sannigrahi et al. (2018) forest class codes.
4. For every fire point, looks up its LULC pixel via exact affine grid math
   and keeps only points that fall on a forest pixel.
5. Saves annual + monthly forest-fire CSVs, a combined CSV, an extraction
   summary, and six summary plots.

Steps 3–4 (the compute bottleneck — a ~124M-pixel grid × up to ~300k fire
points/year × 23 years) run on GPU via [CuPy](https://cupy.dev/), with an
automatic fallback to NumPy (CPU) if no compatible GPU/CUDA install is found.

### Data sources

| Data | Source | Included in repo? |
|---|---|---|
| MODIS fire archive (FIRMS, Collection 6.1) | [firms.modaps.eosdis.nasa.gov](https://firms.modaps.eosdis.nasa.gov/download/) | ❌ (218 MB — download separately) |
| LULC yearly maps (ESA-CCI / C3S, 1992–2022) | [Copernicus Climate Data Store](https://cds.climate.copernicus.eu/datasets/satellite-land-cover) | ❌ (~4.5 GB total — download separately) |
| India state boundary (dissolved for clipping/plotting) | `India_State_Boundary.shp` | ✅ (small enough to track directly) |

Place the downloaded fire archive CSV and LULC ZIPs at the paths configured
in **Step 2** of the notebook (`FIRE_CSV_PATH`, `LULC_DIR`) before running.

### How to run

```bash
pip install -r requirements.txt
jupyter nbconvert --to notebook --execute --inplace "FOREST_FIRE_POINTS_EXTRACTION(INDIA).ipynb"
# or open it in Jupyter/VS Code and run all cells top to bottom
```

GPU acceleration is optional — install the commented-out CUDA packages in
`requirements.txt` if you have an NVIDIA GPU with CUDA 12.x; otherwise the
pipeline runs on CPU automatically, just slower.

### Outputs (`Forest_Fire_Outputs/`)

```
Forest_Fire_Outputs/
├── all_fire_india_merged.csv          # all MODIS fire points clipped to India, 2000–2022
├── all_forest_fires_2000_2022.csv     # combined forest-only fire points
├── extraction_summary.csv             # one row per year — the headline results (tracked)
├── Monthly_BurnedArea_MCD64A1.csv     # MCD64A1.061 monthly burned area, 266 months (tracked)
├── Annual_BurnedArea_vs_FireCount.csv # annual burned area vs. fire count + correlation inputs (tracked)
├── Biswas2025_AnnualFireCount_Comparison.csv  # year-by-year check vs. Biswas et al. (tracked)
├── boundary/                          # dissolved India boundary (GeoPackage)
├── lulc_extracted/                    # extracted yearly LULC NetCDFs
├── forest_fire_points/                # one CSV per year
├── monthly_records/<year>/            # one CSV per year-month
└── plots/                             # summary PNGs incl. BurnedArea_vs_FireCount.png (tracked)
```

Only `extraction_summary.csv`, the three burned-area/Biswas comparison CSVs
above, and `plots/` are tracked in git — everything else is large and
reproducible by re-running the notebook (see `.gitignore`).

### Latest results (23 years, 2000–2022)

- **1,599,466** total MODIS fire points inside India after boundary clip + dedup
- **541,545** of those fall on forest LULC pixels
- Forest cover held steady at **~9.9–10.4%** of India's land area over the period
- Peak year: **2021** (111,467 total fire points, 38,116 forest fires)

| Year | Total fires | Forest fires | Forest fire % | Forest cover % |
|---:|---:|---:|---:|---:|
| 2000 | 1,419 | 225 | 15.86 | 9.86 |
| 2001 | 18,844 | 6,007 | 31.88 | 9.95 |
| 2002 | 26,939 | 4,290 | 15.92 | 9.95 |
| 2003 | 56,119 | 22,348 | 39.82 | 9.97 |
| 2004 | 64,757 | 28,668 | 44.27 | 10.01 |
| 2005 | 63,737 | 22,270 | 34.94 | 10.00 |
| 2006 | 66,195 | 27,241 | 41.15 | 10.00 |
| 2007 | 75,413 | 29,008 | 38.47 | 10.01 |
| 2008 | 71,025 | 23,356 | 32.88 | 10.01 |
| 2009 | 90,500 | 40,155 | 44.37 | 10.01 |
| 2010 | 76,894 | 31,108 | 40.46 | 10.01 |
| 2011 | 72,441 | 22,513 | 31.08 | 10.02 |
| 2012 | 93,536 | 35,214 | 37.65 | 9.91 |
| 2013 | 71,219 | 22,245 | 31.23 | 10.02 |
| 2014 | 76,638 | 23,844 | 31.11 | 10.02 |
| 2015 | 68,553 | 20,258 | 29.55 | 10.02 |
| 2016 | 89,354 | 28,903 | 32.35 | 10.13 |
| 2017 | 82,729 | 26,462 | 31.99 | 10.17 |
| 2018 | 91,342 | 29,028 | 31.78 | 10.19 |
| 2019 | 75,693 | 21,158 | 27.95 | 10.27 |
| 2020 | 76,149 | 16,421 | 21.56 | 10.30 |
| 2021 | 111,467 | 38,116 | 34.19 | 10.33 |
| 2022 | 78,503 | 22,707 | 28.93 | 10.43 |

All 27 study years (2000–2022) now use **real** ESA-CCI/C3S LULC data — no
nearest-year fallback is used anywhere in this range.

### Supplementary validation: burned area vs. fire count (MCD64A1.061)

An independent cross-check against MODIS MCD64A1.061 burned-area, aggregated
to the same annual (2000–2022) grain as this step's own forest-fire-point
counts. Recomputed 2026-08-20 from a complete, single-mosaic AppEEARS
download covering the full study period `2000-11-01`–`2022-12-15` with no
month gaps (an earlier version of this analysis was run on a source dataset
missing January/February every year and had to restrict both series to a
Mar–Dec-matched subset; that limitation is now resolved).

- **Pearson r = 0.915** (p < 0.0001), **Spearman ρ = 0.835** (p < 0.0001),
  n = 23 years (2000–2022; full Jan–Dec for 2001–2021, partial for 2000
  [Nov–Dec only, matching the study start] and 2022 [through the Dec 2022
  monthly composite, matching the study's Dec 15 end])
- Files: `Forest_Fire_Outputs/Monthly_BurnedArea_MCD64A1.csv` (266 months),
  `Forest_Fire_Outputs/Annual_BurnedArea_vs_FireCount.csv` (23 years),
  `Forest_Fire_Outputs/plots/BurnedArea_vs_FireCount.png`
- Burned-pixel mask: `(band > 0) & (band <= 366)` (excludes MCD64A1's
  -2=water, -1=nodata/void, 0=unburned fill values); per-pixel area computed
  with a latitude-corrected formula (km² varies with `cos(latitude)` by row),
  not a flat degree×111km conversion.
- A separate year-by-year comparison against Biswas et al. (2025)'s own
  reported annual fire-count percentages (`Biswas2025_AnnualFireCount_
  Comparison.csv`) uses this step's full-year forest-fire-point counts
  directly (independent of the burned-area gap above) and is unaffected by
  this recomputation: this project's counts still run 0.5–2.4% higher than
  Biswas et al.'s derived counts across all 20 overlapping years
  (2001–2020).

### Repo structure

```
.
├── FOREST_FIRE_POINTS_EXTRACTION(INDIA).ipynb  # Step 1 pipeline (this notebook)
├── build_notebook.py                            # script that generates the notebook programmatically
├── India_State_Boundary.shp / .shx               # India boundary used for clipping/plotting
├── Forest_Fire_Outputs/                          # pipeline outputs (partially tracked, see above)
├── requirements.txt
└── .gitignore
```

## Citation

- Biswas, S. et al. (2025). *[Forest fire detection methodology — see notebook
  header for full reference]*
- Uthappa, A. et al. (2025).
- Sannigrahi, S. et al. (2018). ESA-CCI/C3S forest land-cover class mapping.

## License

No license has been chosen yet for this repository's code. The MODIS fire
archive (NASA FIRMS) and ESA-CCI/C3S LULC data are subject to their own
respective data-use terms — see the source links above.
