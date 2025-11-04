# astro_np_pd_plot.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Optional: create the sample CSV if it doesn't exist
csv_path = Path("observations.csv")
if not csv_path.exists():
    csv_text = """time_iso,flux_mjy,flux_err_mjy,band,target
2025-10-01T00:00:00,12.5,0.5,optical,Crab
2025-10-01T00:05:00,13.1,0.6,optical,Crab
2025-10-01T00:10:00,11.8,0.4,optical,Crab
2025-10-01T00:15:00,14.2,0.5,optical,Crab
2025-10-01T00:20:00,50.0,2.0,radio,Crab
2025-10-01T00:25:00,48.5,1.5,radio,Crab
2025-10-01T00:30:00,49.2,1.8,radio,Crab
2025-10-01T00:35:00,47.1,1.6,radio,Crab
2025-10-01T00:40:00,13.0,0.6,optical,Crab
2025-10-01T00:45:00,12.7,0.5,optical,Crab
"""
    csv_path.write_text(csv_text)

# 1) Read CSV with pandas
df = pd.read_csv(csv_path, parse_dates=["time_iso"])
df = df.set_index("time_iso").sort_index()
print("\nDataFrame head:\n", df.head())

# 2) Basic stats using pandas & numpy
# Convert to numpy arrays for a few vectorized ops
flux = df["flux_mjy"].to_numpy()
flux_err = df["flux_err_mjy"].to_numpy()

print("\nNumPy array types:", flux.dtype, flux_err.dtype)
print("Mean flux (mJy):", np.mean(flux))
print("Std dev flux (mJy):", np.std(flux, ddof=1))  # sample std

# 3) Grouped statistics by band (Pandas)
grouped = df.groupby("band")["flux_mjy"].agg(["count", "mean", "std", "min", "max"])
print("\nGrouped stats by band:\n", grouped)

# 4) Weighted mean example using numpy for optical band only
optical = df[df["band"] == "optical"]
w = 1.0 / (optical["flux_err_mjy"].to_numpy() ** 2)
weighted_mean = np.sum(optical["flux_mjy"].to_numpy() * w) / np.sum(w)
print("\nWeighted mean flux (optical):", weighted_mean)

# 5) Time-series resampling example (5-minute -> 10-minute means)
resampled = df["flux_mjy"].resample("10T").mean()
print("\nResampled (10T) means:\n", resampled)

# 6) Plotting: light curves per band (time vs flux)
plt.figure(figsize=(9, 4))
for band, sub in df.groupby("band"):
    plt.errorbar(sub.index, sub["flux_mjy"], yerr=sub["flux_err_mjy"], marker="o", linestyle="-", label=band)
plt.xlabel("Time (UTC)")
plt.ylabel("Flux (mJy)")
plt.title("Light curves by band — Crab (sample)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("lightcurves.png")
print("\nSaved lightcurves.png")

# 7) Histogram of fluxes
plt.figure(figsize=(5, 4))
plt.hist(df["flux_mjy"].to_numpy(), bins=8)
plt.xlabel("Flux (mJy)")
plt.ylabel("Counts")
plt.title("Flux distribution (all bands)")
plt.tight_layout()
plt.savefig("flux_hist.png")
print("Saved flux_hist.png")

# 8) Scatter: flux vs flux_err (quick QA plot)
plt.figure(figsize=(5, 4))
plt.scatter(df["flux_mjy"], df["flux_err_mjy"])
plt.xlabel("Flux (mJy)")
plt.ylabel("Flux error (mJy)")
plt.title("Flux vs Flux error")
plt.tight_layout()
plt.savefig("flux_vs_err.png")
print("Saved flux_vs_err.png")

# 9) Save processed summary to CSV
summary_path = Path("summary_by_band.csv")
grouped.to_csv(summary_path)
print("\nWrote summary_by_band.csv")
