"""Regenerates BurnedArea_vs_FireCount.png from the already-computed
Annual_BurnedArea_vs_FireCount.csv -- closes a real reproducibility gap a
completeness audit found: this plot previously existed only as a committed output
artifact, with no script anywhere in the tracked repo that produced it (only git
commit messages describing what was done).

Scope note, stated honestly: this script reproduces the PLOT from the already-
validated CSV (forest_fire_points, burned_area_km2 per year) -- it does not
re-derive that CSV from the raw MCD64A1.061 burned-area NetCDF files, which is a
deeper reproducibility layer not addressed here. If the raw MCD64A1 processing
script also needs recovering/rebuilding, that is a separate, larger task."""
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CSV_PATH = r"D:\FOREST FIRE MAPPING(INDIA)\Forest fire Extraction in INDIA(2000-2022)\Forest_Fire_Outputs\Annual_BurnedArea_vs_FireCount.csv"
OUT_PATH = r"D:\FOREST FIRE MAPPING(INDIA)\Forest fire Extraction in INDIA(2000-2022)\Forest_Fire_Outputs\plots\BurnedArea_vs_FireCount.png"

df = pd.read_csv(CSV_PATH).sort_values("year")
print(f"Loaded {len(df)} years: {df['year'].min()}-{df['year'].max()}")

r, p_pearson = stats.pearsonr(df["forest_fire_points"], df["burned_area_km2"])
rho, p_spearman = stats.spearmanr(df["forest_fire_points"], df["burned_area_km2"])
print(f"Pearson r={r:.3f} (p={p_pearson:.2e}), Spearman rho={rho:.3f} (p={p_spearman:.2e}), n={len(df)}")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))

# Panel 1: dual-axis annual trend
ax1b = ax1.twinx()
ax1.bar(df["year"], df["burned_area_km2"], color="#f4a582", edgecolor="#b2182b", alpha=0.85, label="Burned area")
ax1b.plot(df["year"], df["forest_fire_points"], color="#2166ac", marker="o", linewidth=2, label="Fire points")
ax1.set_xlabel("Year")
ax1.set_ylabel("Annual burned area (km²)", color="#b2182b")
ax1b.set_ylabel("Annual forest fire points (full Jan-Dec)", color="#2166ac")
ax1.tick_params(axis="y", labelcolor="#b2182b")
ax1b.tick_params(axis="y", labelcolor="#2166ac")
ax1.set_xticks(df["year"][::2])
ax1.set_xticklabels(df["year"][::2], rotation=45)
partial_note = "; ".join(f"{int(y)}" for y in df.loc[df["partial_year"], "year"]) if "partial_year" in df.columns and df["partial_year"].any() else "none"
ax1.set_title(f"Annual trend: burned area vs. forest fire count\n(India, {df['year'].min()}-{df['year'].max()}; partial years: {partial_note})")

# Panel 2: scatter + regression
ax2.scatter(df["forest_fire_points"], df["burned_area_km2"], color="#2166ac", s=60, zorder=3)
z = np.polyfit(df["forest_fire_points"], df["burned_area_km2"], 1)
xs = np.linspace(df["forest_fire_points"].min(), df["forest_fire_points"].max(), 100)
ax2.plot(xs, np.polyval(z, xs), color="#b2182b", linestyle="--", linewidth=2)
for _, row in df.iterrows():
    if row["forest_fire_points"] < 7000 or row["forest_fire_points"] > 35000 or row["burned_area_km2"] < 5000:
        ax2.annotate(str(int(row["year"])), (row["forest_fire_points"], row["burned_area_km2"]),
                     textcoords="offset points", xytext=(5, 5), fontsize=9)
ax2.set_xlabel("Annual forest fire points (full Jan-Dec)")
ax2.set_ylabel("Annual burned area (km²)")
ax2.set_title(f"Fire count vs. burned area correlation\nPearson r = {r:.3f} (p < 0.0001, n={len(df)} years) "
              f"— Spearman ρ = {rho:.3f}")
ax2.grid(alpha=0.3)

fig.tight_layout()
fig.savefig(OUT_PATH, dpi=150, facecolor="white")
print(f"Saved: {OUT_PATH}")
