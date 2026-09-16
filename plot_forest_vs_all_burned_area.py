"""Burned-area comparison plot, two panels: (top) this project's all-land-cover
series, this project's forest-masked series (new), and Biswas et al. (2025)'s own
Fig. 7d burned-area chart, digitized by careful pixel-level reading of the published
figure (no exact table exists in their text, so their underlying data isn't
available); (bottom) the residual gap between the forest-masked series and Biswas et
al., marked explicitly (not left for the reader to infer from the top panel alone) --
both the absolute difference and the ratio."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

OUT_PATH = "Forest_Fire_Outputs/plots/BurnedArea_ForestVsAll_vs_Biswas.png"

annual = pd.read_csv("Forest_Fire_Outputs/Annual_BurnedArea_ForestVsAll.csv")
df = annual[(annual["year"] >= 2001) & (annual["year"] <= 2020)].sort_values("year").reset_index(drop=True)

# Biswas et al. (2025) Fig. 7d, digitized by careful pixel-level reading of the
# published bar chart (their paper gives no exact table) -- approximate, not exact.
biswas_years = list(range(2001, 2021))
biswas_km2 = [2650, 2100, 6450, 9850, 5650, 5700, 9200, 6300, 17200, 12700,
              7400, 16400, 8800, 7900, 6100, 8400, 11500, 10400, 7200, 4400]
df["biswas_fig7d_digitized_km2"] = biswas_km2
df["diff_forest_minus_biswas_km2"] = df["burned_area_km2_forest_only"] - df["biswas_fig7d_digitized_km2"]
df["ratio_forest_vs_biswas"] = df["burned_area_km2_forest_only"] / df["biswas_fig7d_digitized_km2"]

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 9), sharex=True, gridspec_kw={"height_ratios": [1.6, 1]})

ax1.bar(df["year"] - 0.27, df["burned_area_km2_all_landcover"] / 1000, width=0.27,
        color="#4472a8", label="This study — all land cover")
ax1.bar(df["year"], df["burned_area_km2_forest_only"] / 1000, width=0.27,
        color="#4a8a4a", label="This study — forest-masked")
ax1.bar(df["year"] + 0.27, df["biswas_fig7d_digitized_km2"] / 1000, width=0.27,
        color="#c97b2e", label="Biswas et al. (2025), Fig. 7d — digitized")
ax1.set_ylabel("Annual burned area (thousand km²)")
ax1.legend(loc="upper left", fontsize=9.5)
ax1.grid(alpha=0.25, axis="y")
ax1.set_title("MCD64A1.061 annual burned area, 2001–2020: all-land-cover vs. forest-masked\n"
              "vs. Biswas et al. (2025)'s own (forest-scoped) chart", fontsize=12)

# --- Marked-separately difference panel: forest-masked vs. Biswas, explicitly ---
bars = ax2.bar(df["year"], df["diff_forest_minus_biswas_km2"] / 1000, color="#8a4a8a", alpha=0.85,
                label="Forest-masked − Biswas et al. (residual gap)")
for x, y, r in zip(df["year"], df["diff_forest_minus_biswas_km2"] / 1000, df["ratio_forest_vs_biswas"]):
    ax2.annotate(f"{r:.1f}×", (x, y + 0.6), ha="center", fontsize=7.5, color="#5a2a5a")
ax2.set_ylabel("Residual gap (thousand km²)")
ax2.set_xlabel("Year")
ax2.set_xticks(df["year"])
ax2.set_xticklabels(df["year"], rotation=45)
ax2.grid(alpha=0.25, axis="y")
ax2.legend(loc="upper left", fontsize=9.5)
ax2.set_title("Residual gap after forest-masking, forest-masked − Biswas et al. (label: ratio ×)",
              fontsize=10.5)

fig.tight_layout()
fig.savefig(OUT_PATH, dpi=150, facecolor="white")
print(f"Saved: {OUT_PATH}")

print("\nyear  all_landcover  forest_only  biswas_digitized  diff(forest-biswas)  ratio")
for _, row in df.iterrows():
    print(f"{int(row['year'])}  {row['burned_area_km2_all_landcover']:>13.0f}  "
          f"{row['burned_area_km2_forest_only']:>11.0f}  {row['biswas_fig7d_digitized_km2']:>16.0f}  "
          f"{row['diff_forest_minus_biswas_km2']:>19.0f}  {row['ratio_forest_vs_biswas']:>5.2f}×")
