"""
Recalculation R1 -- fire labels, from the RAW MODIS C6.1 FIRMS archive.

Independent re-implementation of Step 1 (does not import or execute the notebook):
  raw archive -> valid rows -> India bbox -> exact India polygon -> study period ->
  duplicate removal -> forest filter against the SAME-YEAR ESA-CCI/C3S LCCS map.
Then: set-level comparison with the historical 541,545-point CSV, FIRMS confidence /
type statistics, label-sensitivity variants, boundary sensitivity (state vs country vs
GADM), rasterisation to the NDVI grid (explains any points lost), India-only forest
cover (the historical 'forest_cover_pct' was computed over the download rectangle),
and the Biswas et al. annual-count comparison re-derived from their published numbers.
"""
import glob
import os
import re
import time

import netCDF4
import numpy as np
import pandas as pd
import shapely

from audit_common import (P, RES, FOREST_CODES, STUDY_START, STUDY_END, provenance, dump,
                          ndvi_grid, india_geom, india_mask, rowcol, save_tif)

OUT = os.path.join(RES, "recalculated", "R1_fire_labels")
os.makedirs(OUT, exist_ok=True)
t0 = time.time()
rep = dict(provenance=provenance(__file__))

# ------------------------------------------------------------------ stage counts
df = pd.read_csv(P["fire_raw"], low_memory=False)
df.columns = [c.strip().lower() for c in df.columns]
st = dict(raw_archive=len(df))
df["acq_date"] = pd.to_datetime(df["acq_date"], format="%Y-%m-%d", errors="coerce")
df = df.dropna(subset=["acq_date", "latitude", "longitude"]); st["valid_rows"] = len(df)
df = df[(df.longitude >= 68.0) & (df.longitude <= 97.5) & (df.latitude >= 6.5) & (df.latitude <= 37.5)]
st["india_bbox"] = len(df)
geoms = {w: india_geom(w) for w in ("state", "country", "gadm")}
inside = {w: shapely.contains_xy(g, df.longitude.values, df.latitude.values) for w, g in geoms.items()}
rep["boundary_sensitivity_points_in_bbox"] = {w: int(v.sum()) for w, v in inside.items()}
rep["boundary_sensitivity_disagreement"] = dict(
    state_not_country=int((inside["state"] & ~inside["country"]).sum()),
    country_not_state=int((~inside["state"] & inside["country"]).sum()),
    state_not_gadm=int((inside["state"] & ~inside["gadm"]).sum()),
    gadm_not_state=int((~inside["state"] & inside["gadm"]).sum()))
df_all_bbox = df.copy()
df_all_bbox["in_state"], df_all_bbox["in_gadm"] = inside["state"], inside["gadm"]
df = df[inside["state"]]; st["india_polygon_state"] = len(df)
df = df[(df.acq_date >= STUDY_START) & (df.acq_date <= STUDY_END)]; st["study_period"] = len(df)
df = df.drop_duplicates(subset=["longitude", "latitude", "acq_date"]); st["dedup_lon_lat_date"] = len(df)
df["year"], df["month"] = df.acq_date.dt.year, df.acq_date.dt.month

# ------------------------------------------------------------------ forest filter per year
lulc_files = {}
for f in glob.glob(os.path.join(P["lulc_dir"], "*.nc")):
    m = re.search(r"P1Y-(\d{4})", os.path.basename(f))
    lulc_files[int(m.group(1))] = f
years = sorted(df.year.unique())
rep["lulc_years_missing_in_study_period"] = [int(y) for y in years if y not in lulc_files]

forest_flag = np.zeros(len(df), dtype=bool)
lulc_class = np.full(len(df), -1, dtype=np.int16)
cover = []
gm = india_geom("state")
for y in years:
    sel = np.where(df.year.values == y)[0]
    with netCDF4.Dataset(lulc_files[y]) as ds:
        lat = np.asarray(ds["lat"][:]); lon = np.asarray(ds["lon"][:])
        lc = np.asarray(ds["lccs_class"][0] if ds["lccs_class"].ndim == 3 else ds["lccs_class"][:]).astype(np.int16)
    dlat, dlon = lat[1] - lat[0], lon[1] - lon[0]
    # pixel containing the point (edges = centres -/+ half a pixel)
    r = np.floor((df.latitude.values[sel] - (lat[0] - dlat / 2)) / dlat).astype(int)
    c = np.floor((df.longitude.values[sel] - (lon[0] - dlon / 2)) / dlon).astype(int)
    r = np.clip(r, 0, len(lat) - 1); c = np.clip(c, 0, len(lon) - 1)
    lulc_class[sel] = lc[r, c]
    forest_flag[sel] = np.isin(lc[r, c], FOREST_CODES)
    fmask = np.isin(lc, FOREST_CODES)
    # India-only forest cover (pixel-centre in polygon), sampled every 4th pixel for tractability
    LA, LO = np.meshgrid(lat[::4], lon[::4], indexing="ij")
    ins = shapely.contains_xy(gm, LO.ravel(), LA.ravel()).reshape(LA.shape)
    cover.append(dict(year=int(y), file=os.path.basename(lulc_files[y]),
                      forest_pct_download_rectangle=100 * fmask.mean(),
                      forest_pct_india_only=100 * fmask[::4, ::4][ins].mean(),
                      total_points=len(sel), forest_points=int(forest_flag[sel].sum())))
    print(y, len(sel), int(forest_flag[sel].sum()), flush=True)
df["lulc_class"] = lulc_class
ff = df[forest_flag].copy()
st["forest_filter"] = len(ff)
rep["stage_counts"] = st
pd.DataFrame(cover).to_csv(os.path.join(OUT, "annual_extraction_and_forest_cover.csv"), index=False)

# ------------------------------------------------------------------ compare with historical CSV
hist = pd.read_csv(P["fire_hist"])
key = ["latitude", "longitude", "acq_date", "acq_time", "satellite"]
ff_k = ff.assign(acq_date=ff.acq_date.dt.strftime("%Y-%m-%d"))[key]
a = set(map(tuple, ff_k.values.tolist())); b = set(map(tuple, hist[key].values.tolist()))
rep["historical_csv_comparison"] = dict(recalc=len(a), historical=len(b), common=len(a & b),
                                        recalc_only=len(a - b), historical_only=len(b - a))
if a - b:
    pd.DataFrame(list(a - b), columns=key).to_csv(os.path.join(OUT, "recalc_only_points.csv"), index=False)
if b - a:
    pd.DataFrame(list(b - a), columns=key).to_csv(os.path.join(OUT, "historical_only_points.csv"), index=False)

# ------------------------------------------------------------------ FIRMS QA fields
rep["firms_fields"] = dict(
    confidence_lt30=int((ff.confidence < 30).sum()), confidence_lt30_pct=100 * (ff.confidence < 30).mean(),
    confidence_quantiles={q: float(ff.confidence.quantile(q)) for q in (0.05, 0.25, 0.5, 0.75)},
    type_counts={int(k): int(v) for k, v in ff["type"].value_counts().items()},
    type_nonzero_pct=100 * (ff["type"] != 0).mean(),
    satellite={k: int(v) for k, v in ff.satellite.value_counts().items()},
    daynight={k: int(v) for k, v in ff.daynight.value_counts().items()},
    exact_duplicate_rows=int(ff.duplicated().sum()),
    same_lat_lon_different_day=int(ff.duplicated(["latitude", "longitude"]).sum()))

# ------------------------------------------------------------------ rasterise to the NDVI grid
g = ndvi_grid()
H, W = g["shape"]
imask = india_mask(g)
ndvi_valid_hist = np.isfinite(__import__("audit_common").read_tif(os.path.join(P["ndvi_out"], "F1_NDVI_QA_mean.tif"))[0])
variants = {
    "all": np.ones(len(ff), bool),
    "conf_ge30": ff.confidence.values >= 30,
    "type0": ff["type"].values == 0,
    "conf_ge30_type0": (ff.confidence.values >= 30) & (ff["type"].values == 0),
    "years_2001_2020": (ff.year.values >= 2001) & (ff.year.values <= 2020),
    "year_2020_only": ff.year.values == 2020,
}
r, c = rowcol(g["transform"], ff.longitude.values, ff.latitude.values)
inb = (r >= 0) & (r < H) & (c >= 0) & (c < W)
rep["rasterisation"] = dict(points=len(ff), out_of_grid=int((~inb).sum()),
                            outside_india_mask=int((inb & ~imask[np.clip(r, 0, H - 1), np.clip(c, 0, W - 1)]).sum()),
                            outside_ndvi_valid=int((inb & ~ndvi_valid_hist[np.clip(r, 0, H - 1), np.clip(c, 0, W - 1)]).sum()))
labels = {}
for name, sel in variants.items():
    s = sel & inb
    cnt = np.zeros((H, W), np.int32)
    np.add.at(cnt, (r[s], c[s]), 1)
    labels[name] = cnt
    rep.setdefault("label_variants", {})[name] = dict(
        points=int(sel.sum()), fire_pixels_grid=int((cnt > 0).sum()),
        fire_pixels_in_ndvi_valid=int(((cnt > 0) & ndvi_valid_hist).sum()),
        points_in_ndvi_valid=int(cnt[ndvi_valid_hist].sum()))
np.savez_compressed(os.path.join(OUT, "fire_count_variants_ndvi_grid.npz"), **labels)
save_tif(labels["all"].astype(np.float32), os.path.join(OUT, "fire_count_all_recalc.tif"))

# historical F10 raster comparison
from audit_common import read_tif, compare
f10, _ = read_tif(os.path.join(P["ndvi_out"], "F10_fire_count_Step1.tif"))
rep["F10_comparison"] = compare(labels["all"], f10, name="fire_count vs F10_fire_count_Step1.tif")
rep["F10_comparison"]["hist_sum"] = float(np.nansum(f10)); rep["F10_comparison"]["recalc_sum"] = int(labels["all"].sum())

# ------------------------------------------------------------------ monthly declustering statistic
pix_day = pd.Series(list(zip(r, c, ff.acq_date.values)))
rep["same_pixel_same_day_multiple_detections"] = int(pix_day.duplicated().sum())

# ------------------------------------------------------------------ Biswas annual counts (published numbers only)
biswas_pct = {2009: 8.36, 2012: 7.30, 2010: 6.48, 2007: 6.05, 2018: 6.02, 2004: 5.98, 2016: 5.98, 2006: 5.67,
              2017: 5.51, 2014: 4.97, 2008: 4.87, 2011: 4.69, 2003: 4.66, 2013: 4.65, 2005: 4.62, 2019: 4.41,
              2015: 4.23, 2020: 3.42, 2001: 1.24, 2002: 0.88}   # Biswas et al. 2025, p.4865
tot = 232189 + 243761                                              # their decade totals, p.4865
ours = ff.groupby("year").size()
rows = []
for y in sorted(biswas_pct):
    est = biswas_pct[y] / 100 * tot
    lo, hi = (biswas_pct[y] - 0.005) / 100 * tot, (biswas_pct[y] + 0.005) / 100 * tot
    rows.append(dict(year=y, biswas_pct=biswas_pct[y], biswas_count_derived=round(est),
                     biswas_count_rounding_lo=round(lo), biswas_count_rounding_hi=round(hi),
                     this_study=int(ours.get(y, 0)), diff_pct=100 * (ours.get(y, 0) - est) / est))
bdf = pd.DataFrame(rows)
bdf.to_csv(os.path.join(OUT, "biswas_annual_count_comparison_recalc.csv"), index=False)
rep["biswas_counts"] = dict(sum_of_pcts=sum(biswas_pct.values()), decade_totals_sum=tot,
                            this_study_2001_2020=int(ours.loc[2001:2020].sum()),
                            diff_pct_range=[float(bdf.diff_pct.min()), float(bdf.diff_pct.max())],
                            pearson_r=float(np.corrcoef(bdf.biswas_count_derived, bdf.this_study)[0, 1]))
rep["annual_counts"] = {int(k): int(v) for k, v in ours.items()}
rep["runtime_sec"] = time.time() - t0
ff.to_parquet(os.path.join(OUT, "forest_fire_points_recalc.parquet"), index=False)
dump(rep, os.path.join(OUT, "R1_report.json"))
print({k: v for k, v in rep.items() if k in ("stage_counts", "historical_csv_comparison", "rasterisation", "firms_fields", "F10_comparison", "biswas_counts")})
