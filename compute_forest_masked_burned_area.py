"""Re-derives MCD64A1.061 burned area TWO ways from the raw AppEEARS GeoTIFFs in
"../Burn Area Data @INDIA/Geotiff Files_@INDIA/": (1) all-land-cover (replicates the
existing Monthly_BurnedArea_MCD64A1.csv methodology, as a validation check), and
(2) forest-only (new -- applies Step 1's own FOREST_CODES mask, per-calendar-year,
via the same affine pixel-lookup convention build_notebook.py uses for fire points).

Motivation: Biswas et al. (2025)'s Fig. 7d burned-area chart is forest-scoped (their
whole paper is about forest fires); this project's existing burned-area CSV was never
forest-masked (all India, every land-cover type), which explained a ~6-10x magnitude
gap despite both series tracking the same interannual pattern (see
Step1_FirePointExtraction_Audit_and_Documentation.md, 2026-09-16 addition). This script
closes that gap by adding a genuinely forest-masked series for a real apples-to-apples
comparison.

Forest mask: build_notebook.py's FOREST_CODES = {50,60,61,62,70,71,72,80,81,82,90,100,110}
(ESA-CCI/C3S LCCS codes), per-calendar-year LULC file, same nearest-year fallback for
missing years (2000, 2012) as the main pipeline.

Burned-pixel mask: (band > 0) & (band <= 366), matching the documented convention in
this repo's own README (excludes -2=water, -1=nodata/void, 0=unburned fill).

Per-pixel area: latitude-corrected (km^2 shrinks with cos(lat) by row), not a flat
degree*111km conversion -- matching the existing methodology's own documented approach.
"""
import os
import re
import glob
import numpy as np
import pandas as pd
import rasterio
import netCDF4 as nc_lib
from tqdm import tqdm

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BURN_DIR = os.path.join(BASE_DIR, "..", "Burn Area Data @INDIA", "Geotiff Files_@INDIA")
LULC_DIR = os.path.join(BASE_DIR, "Forest_Fire_Outputs", "lulc_extracted")
OUT_CSV = os.path.join(BASE_DIR, "Forest_Fire_Outputs", "Monthly_BurnedArea_ForestVsAll.csv")

START_YEAR, END_YEAR = 2000, 2022
FOREST_CODES = {50, 60, 61, 62, 70, 71, 72, 80, 81, 82, 90, 100, 110}
EARTH_KM_PER_DEG = 111.32

# ---------------------------------------------------------------------- #
# 1. Build the LULC year map -- same nearest-year fallback as build_notebook.py
# ---------------------------------------------------------------------- #
lulc_files = glob.glob(os.path.join(LULC_DIR, "*.nc"))
lulc_year_map = {}
for f in lulc_files:
    m = re.search(r"(19|20)\d{2}", os.path.basename(f))
    if not m:
        continue
    yr = int(m.group())
    if yr > END_YEAR:
        continue
    lulc_year_map[yr] = f

needed = set(range(START_YEAR, END_YEAR + 1))
available_yrs = sorted(lulc_year_map.keys())
missing = sorted(needed - set(available_yrs))
for y in missing:
    nearest = min(available_yrs, key=lambda x: abs(x - y))
    lulc_year_map[y] = lulc_year_map[nearest]
print(f"LULC years available: {available_yrs}")
print(f"LULC years filled via nearest-year fallback: {missing}")

# ---------------------------------------------------------------------- #
# 2. Burn-date GeoTIFF file list, parsed into (year, month)
# ---------------------------------------------------------------------- #
burn_files = sorted(glob.glob(os.path.join(BURN_DIR, "MCD64A1.061_Burn_Date_doy*_aid0001.tif")))
burn_files = [f for f in burn_files if "Uncertainty" not in f]
print(f"Burn_Date GeoTIFFs found: {len(burn_files)}")


def parse_doy_filename(path):
    m = re.search(r"doy(\d{4})(\d{3})", os.path.basename(path))
    year, doy = int(m.group(1)), int(m.group(2))
    date = pd.Timestamp(f"{year}-01-01") + pd.Timedelta(days=doy - 1)
    return year, date.month


# ---------------------------------------------------------------------- #
# 3. Burn grid geometry (identical for every month -- read once)
# ---------------------------------------------------------------------- #
with rasterio.open(burn_files[0]) as src:
    H, W = src.shape
    transform = src.transform
    burn_crs = src.crs

lon0 = transform.c + transform.a / 2.0
dlon_burn = transform.a
lat0 = transform.f + transform.e / 2.0
dlat_burn = transform.e  # negative (north-down)

burn_lons = lon0 + dlon_burn * np.arange(W)
burn_lats = lat0 + dlat_burn * np.arange(H)

# Per-row pixel area (km^2), latitude-corrected
lat_rad = np.radians(burn_lats)
pixel_area_row_km2 = (abs(dlat_burn) * EARTH_KM_PER_DEG) * (abs(dlon_burn) * EARTH_KM_PER_DEG * np.cos(lat_rad))
pixel_area_grid = np.broadcast_to(pixel_area_row_km2[:, None], (H, W))
print(f"Burn grid: {H}x{W}, resolution ~{abs(dlat_burn)*111.32*1000:.0f}m, "
      f"pixel area range {pixel_area_row_km2.min():.4f}-{pixel_area_row_km2.max():.4f} km2")


def build_forest_mask_on_burn_grid(nc_path):
    """Nearest-neighbor lookup of each burn-grid pixel center against the LULC's
    own native grid, same affine-math convention as build_notebook.py's
    points_to_pixel_indices() (applied to a full raster of centers here, not points)."""
    with nc_lib.Dataset(nc_path, "r") as ds:
        lulc = np.array(ds.variables["lccs_class"][:])
        lulc_lat = np.array(ds.variables["lat"][:])
        lulc_lon = np.array(ds.variables["lon"][:])
    if lulc.ndim == 3:
        lulc = lulc[0]
    forest_bool = np.isin(lulc, sorted(FOREST_CODES))

    dlat_l = float(lulc_lat[1] - lulc_lat[0])
    dlon_l = float(lulc_lon[1] - lulc_lon[0])
    row_l = np.round((burn_lats - lulc_lat[0]) / dlat_l).astype(np.int32)
    col_l = np.round((burn_lons - lulc_lon[0]) / dlon_l).astype(np.int32)
    row_l = np.clip(row_l, 0, len(lulc_lat) - 1)
    col_l = np.clip(col_l, 0, len(lulc_lon) - 1)

    return forest_bool[row_l[:, None], col_l[None, :]]


# ---------------------------------------------------------------------- #
# 4. Precompute one forest mask per calendar year (reused across its months)
# ---------------------------------------------------------------------- #
print("\nBuilding per-year forest masks on the burn grid...")
forest_masks_by_year = {}
for yr in tqdm(sorted(set(needed)), desc="Years"):
    forest_masks_by_year[yr] = build_forest_mask_on_burn_grid(lulc_year_map[yr])
    fpct = 100.0 * forest_masks_by_year[yr].sum() / forest_masks_by_year[yr].size
    tqdm.write(f"  {yr}: forest cover on burn grid = {fpct:.2f}%")

# ---------------------------------------------------------------------- #
# 5. Process every month: all-land-cover AND forest-only burned area
# ---------------------------------------------------------------------- #
rows = []
for f in tqdm(burn_files, desc="Months"):
    year, month = parse_doy_filename(f)
    with rasterio.open(f) as src:
        band = src.read(1)
    burned = (band > 0) & (band <= 366)
    forest_mask = forest_masks_by_year[year]

    area_all = float(pixel_area_grid[burned].sum())
    area_forest = float(pixel_area_grid[burned & forest_mask].sum())
    n_all = int(burned.sum())
    n_forest = int((burned & forest_mask).sum())

    rows.append(dict(year=year, month=month, n_burned_pixels_all=n_all,
                      burned_area_km2_all_landcover=area_all,
                      n_burned_pixels_forest=n_forest,
                      burned_area_km2_forest_only=area_forest))

df = pd.DataFrame(rows).sort_values(["year", "month"]).reset_index(drop=True)
df.to_csv(OUT_CSV, index=False)
print(f"\nSaved: {OUT_CSV}")
print(df.head(10))
