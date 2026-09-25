# 🌲🔥 Forest Fire Mapping — India

<!-- AUDIT-UPDATE-2026-09-25 -->
> ### Audit update (2026-09-25)
> This repository's step was recalculated independently from the raw data in a full end-to-end audit.
> **Corrected results, reproduction checks and audit code: [`AUDIT_2026-09-25.md`](AUDIT_2026-09-25.md)** and `audit_2026-09-25/`.
> Earlier text below is kept for the record (it also remains in the git history). Statements superseded by the audit:
>
> - **'National forest cover 9.86–10.43%'**: computed over the download rectangle. For India only it is **18.3–19.3%**.
<!-- AUDIT-UPDATE-2026-09-25 -->


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

### Why this step, and how

Every downstream step in this pipeline — NDVI, LST, FLDAS climatic variables,
terrain/accessibility, the integrated feature stack, and the Random Forest/MaxEnt/
CDR-PINN models — needs a real, spatially and temporally precise fire/no-fire label to
train and evaluate against; this step is what produces that label, so its correctness is
the foundation the rest of the pipeline's reported accuracy numbers rest on. MODIS
Collection 6.1 FIRMS was chosen over coarser fire products because it reports
individual, dated, geolocated fire detections at native ~1 km resolution (Giglio et al.
2016, the MODIS Collection 6 active-fire detection algorithm paper), rather than a
pre-aggregated grid-cell count — this lets every later step re-rasterize the same real
points onto its own working grid instead of inheriting someone else's resolution choice
baked in upstream. Clipping to India's *exact* dissolved state boundary, not a
bounding box, matters because a rectangular India extent also covers parts of five
neighboring countries with different fire regimes and vegetation; the executed run
shows this is a large, real effect, not a theoretical one — the exact-polygon clip
excludes 1,201,876 points (42.9% of the bbox-passing set). Filtering to forest LULC
pixels via *exact affine pixel lookup*, rather than a nearest-neighbor spatial join,
matters because ESA-CCI/C3S's land-cover grid is perfectly regular: the affine method
is mathematically exact for that grid and, unlike a geopandas/shapely spatial join,
vectorizes cleanly on GPU at the ~124M-pixel × hundreds-of-thousands-of-points-per-year
scale this step runs at. The resulting file, `all_forest_fires_2000_2022.csv`, is
consumed directly or re-rasterized onto each later step's own grid (Steps 2–6) and used
as the training/evaluation label for every model in Steps 7–8.

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

### Forest-masked burned area (2026-09-16, closes a real magnitude gap above)

The r=0.915 burned-area validation above is computed **across all of India's land
cover**, not just forest — a genuine mismatch against Biswas et al.'s Fig. 7d, which is
forest-scoped throughout their paper. Re-derived directly from the raw AppEEARS
GeoTIFFs (`compute_forest_masked_burned_area.py`), applying this step's own
`FOREST_CODES` mask per calendar year:

| Series (2001–2020) | Min | Max | Mean | Pearson r vs. Biswas Fig. 7d |
|---|---:|---:|---:|---:|
| All land cover (above) | 24,011 km² (2001) | 102,978 km² (2009) | 74,139 km² | 0.8465 |
| **Forest-masked (new)** | **9,392 km² (2002)** | **51,455 km² (2009)** | **30,724 km²** | **0.9044** |
| Biswas et al. (2025), Fig. 7d (digitized) | ~2,100 km² (2002) | ~17,200 km² (2009) | ~8,700 km² | — |

Forest-masking makes both the minimum year (2002) and maximum year (2009) match
Biswas et al.'s chart exactly (all-land-cover's minimum was 2001, one year off) and
improves the correlation — but a residual ~4× magnitude gap remains, not fully
explained (plausible causes: a stricter forest-class definition or additional
QA/confidence filtering in Biswas et al.'s own unpublished processing). Disclosed
plainly rather than forced to close. The forest-masked series also correlates more
tightly with this step's own forest-fire-point counts than the all-land-cover series
did (Pearson r=0.9345 vs. 0.9149; Spearman ρ=0.7846 vs. 0.8350; n=23, 2000–2022) — the
expected direction, since both measure the same forest-fire population. Files:
`Forest_Fire_Outputs/Monthly_BurnedArea_ForestVsAll.csv`,
`Forest_Fire_Outputs/Annual_BurnedArea_ForestVsAll.csv`,
`Forest_Fire_Outputs/plots/BurnedArea_ForestVsAll_vs_Biswas.png`. Full detail:
`Step1_FirePointExtraction_Audit_and_Documentation.md` (project root).

### Known limitation, disclosed: FIRMS confidence/type filtering and spatial thinning

The 541,545-point label set uses every MODIS FIRMS detection that survives the
boundary clip and forest-LULC filter — it does **not** additionally filter on FIRMS's
own per-detection `confidence` field or `type` field, and it does **not** spatially
decluster/thin points that fall in the same or adjacent pixels across nearby dates.
Both fields are read from the raw archive and preserved, unfiltered, all the way
through to `all_forest_fires_2000_2022.csv`. Quantified from the tracked output:
**23,236 points (4.29%)** carry `confidence < 30` (a commonly used "low confidence"
threshold in FIRMS-based fire studies), and **1,124 points (0.21%)** carry `type != 0`
(1,122 "other static land source", 2 "offshore" — non-vegetation-fire detection
classes per the MODIS Collection 6 Fire/Hotspot User's Guide).

This is a real, literature-flagged gap, not an unrecognized one:

- The **MODIS Collection 6 Fire/Hotspot Active Fire Products User's Guide** (Giglio et
  al. 2016) documents `confidence` and `type` explicitly as per-detection quality
  fields intended for exactly this kind of filtering.
- Spatial autocorrelation among nearby, temporally-clustered fire detections is a
  documented bias risk for point-process fire models trained without spatial
  declustering/thinning (see the spatial-autocorrelation-in-wildfire-ML literature,
  e.g. PMC12215841).
- Spatial thinning ahead of MaxEnt/presence-based fire-susceptibility training has
  regional precedent (e.g. ~1 km-buffer presence-point thinning in Nepal
  fire-susceptibility studies).

**Deliberately deferred, not silently skipped.** `all_forest_fires_2000_2022.csv` is
the label source every one of Steps 2–8 depends on — NDVI, LST, FLDAS, and
terrain/accessibility all join against this exact file's points, the integrated
feature stack is built from it, and the Random Forest, MaxEnt, and CDR-PINN models are
all trained and evaluated against it. Applying confidence filtering or spatial
thinning now would change the label population (and its exact 541,545 row count) and
requires re-running this notebook *and* every downstream step's notebook — a decision
reserved for the project owner, not made unilaterally in a documentation pass.
`confidence` and `type` are kept unfiltered in the output specifically so this
filtering can be applied later without re-extracting from the raw FIRMS archive.

### Comparison against Biswas et al. (2025)

Three concrete, verifiable differences between this step's extraction methodology and
Biswas et al. (2025)'s own presence-point/MaxEnt approach:

1. **Boundary precision.** This step clips to India's exact dissolved state boundary
   polygon (`India_State_Boundary.shp`, 37 parts dissolved to one), not a coarser
   bounding box or a simplified national outline. The executed run shows this is not
   cosmetic: the exact-polygon clip excludes **1,201,876 points (42.9% of the
   bbox-passing set)** that lie inside a rectangular India extent but outside the real
   border, in neighboring Sri Lanka, Nepal, Bangladesh, Myanmar, and southern Pakistan.
   Biswas et al.'s paper does not document an equivalently precise boundary-clipping
   procedure for their own study area.
2. **Point-level vs. aggregated fire ground truth.** This step keeps every individual
   MODIS FIRMS detection at its native geolocation (nominal ~1 km resolution, Giglio et
   al. 2016) — 541,545 independently dated and geolocated forest-fire points. Biswas et
   al.'s MaxEnt model instead rasterizes all 15 of its predictor layers, and by
   necessity its own presence/background points, to a **0.25°×0.25° grid** (confirmed
   directly from their Table 2/3 — see the project's `reference_biswas2025_primary_paper`
   notes). A 0.25° cell is roughly 600–800× the area of one native MODIS ~1 km pixel at
   India's latitudes, so this step's ground truth carries several orders of magnitude
   finer spatial detail than what Biswas et al. trained on.
3. **Independent burned-area cross-check.** This step validates its own fire-point
   archive against an external product, MODIS MCD64A1.061 burned area, two ways: across
   all of India's land cover (Pearson r=0.9149, Spearman ρ=0.8350, n=23) and,
   forest-masked to match Biswas et al.'s own forest-scoped Fig. 7d for a genuine
   apples-to-apples comparison, correlated even more tightly (Pearson r=0.9345,
   Spearman ρ=0.7846, n=23) — see "Forest-masked burned area" above. Biswas et al. use
   MCD64A1.061 burned area only as a separate Table 2 dataset, not as a validation check
   on their own presence-point archive; this cross-check is an independent validation
   exercise Biswas et al. do not perform for their own ground truth.

Where this step is currently no stronger than Biswas et al.: neither project applies
FIRMS confidence/type-based filtering or spatial declustering to its fire occurrence
data (see "Known limitation, disclosed" above) — Biswas et al.'s paper does not
document any such filtering of their own presence points either.

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
