"""Three-way burned-area comparison plot: this project's all-land-cover series,
this project's forest-masked series (new), and Biswas et al. (2025)'s own Fig. 7d
burned-area chart, digitized by careful pixel-level reading of the published figure
(no exact table exists in their text) since their underlying data isn't available."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

OUT_PATH = "Forest_Fire_Outputs/plots/BurnedArea_ForestVsAll_vs_Biswas.png"

annual = pd.read_csv("Forest_Fire_Outputs/Annual_BurnedArea_ForestVsAll.csv")
df = annual[(annual["year"] >= 2001) & (annual["year"] <= 2020)].sort_values("year")

# Biswas et al. (2025) Fig. 7d, digitized by careful pixel-level reading of the
# published bar chart (their paper gives no exact table) -- approximate, not exact.
biswas_years = list(range(2001, 2021))
biswas_km2 = [2650, 2100, 6450, 9850, 5650, 5700, 9200, 6300, 17200, 12700,
              7400, 16400, 8800, 7900, 6100, 8400, 11500, 10400, 7200, 4400]

fig, ax = plt.subplots(figsize=(13, 6))
ax.bar(df["year"] - 0.27, df["burned_area_km2_all_landcover"] / 1000, width=0.27,
       color="#4472a8", label="This study — all land cover")
ax.bar(df["year"], df["burned_area_km2_forest_only"] / 1000, width=0.27,
       color="#4a8a4a", label="This study — forest-masked")
ax.bar(df["year"] + 0.27, [v / 1000 for v in biswas_km2], width=0.27,
       color="#c97b2e", label="Biswas et al. (2025), Fig. 7d — digitized")

ax.set_ylabel("Annual burned area (thousand km²)")
ax.set_xlabel("Year")
ax.set_xticks(df["year"])
ax.set_xticklabels(df["year"], rotation=45)
ax.legend(loc="upper left", fontsize=9.5)
ax.grid(alpha=0.25, axis="y")
ax.set_title("MCD64A1.061 annual burned area, 2001–2020: all-land-cover vs. forest-masked\n"
              "vs. Biswas et al. (2025)'s own (forest-scoped) chart", fontsize=12)

fig.tight_layout()
fig.savefig(OUT_PATH, dpi=150, facecolor="white")
print(f"Saved: {OUT_PATH}")

# Print the numeric comparison table
print("\nyear  all_landcover  forest_only  biswas_digitized")
for i, row in df.iterrows():
    y = int(row["year"])
    b = biswas_km2[biswas_years.index(y)]
    print(f"{y}  {row['burned_area_km2_all_landcover']:>13.0f}  {row['burned_area_km2_forest_only']:>11.0f}  {b:>16}")
