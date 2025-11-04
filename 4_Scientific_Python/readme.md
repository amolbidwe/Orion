# 🛰️ Scientific Python for Astronomy

### *(NumPy • Pandas • Matplotlib)*

Using sample Crab Pulsar flux data

This worksheet demonstrates the **core scientific Python stack** used in astronomy & astrophysics:

| Library        | Purpose                                                       |
| -------------- | ------------------------------------------------------------- |
| **NumPy**      | Fast numerical arrays & vectorized scientific computation     |
| **Pandas**     | Data tables, time-series, CSV ingestion, grouping, resampling |
| **Matplotlib** | Visualizing light curves, histograms, QA plots                |
| **Pathlib**    | Friendly OS-independent file handling                         |

We work with a simulated file of Crab Pulsar optical & radio flux measurements.

---

## 📁 Dataset: `observations.csv`

Automatically created if not present:

| time_iso  | flux_mjy | flux_err_mjy | band            | target |
| --------- | -------: | -----------: | --------------- | ------ |
| Timestamp |     Flux |        Error | optical / radio | Crab   |

Represents **flux vs time** — like a simplified AstroSat / VLA reduced dataset.

> Units: **mJy** (milli-Jansky) — standard radio/optical flux unit

---

## 📦 Code Overview — `astro_np_pd_plot.py`

### ✅ File Creation & Loading (`Path` + `Pandas`)

* Creates CSV if missing
* Reads CSV with time parsing
* Sets timestamp as index → useful for **light-curve work**

```python
df = pd.read_csv(csv_path, parse_dates=["time_iso"])
df = df.set_index("time_iso").sort_index()
```

🔁 `parse_dates` + `.set_index` = time-series ready DataFrame.

---

## 🧠 NumPy — Core Scientific Engine

Used for:

* Fast vector math
* Weighted mean
* Standard deviation
* Converting Pandas columns to arrays

```python
flux = df["flux_mjy"].to_numpy()
np.mean(flux)
np.std(flux, ddof=1)
```

Weighted mean (important in astronomy photometry):

```python
w = 1/(err^2)
weighted_mean = sum(flux*w)/sum(w)
```

---

## 📊 Pandas — Astronomical Table Operations

### 👉 Group by observation band

```python
grouped = df.groupby("band")["flux_mjy"].agg(["count","mean","std","min","max"])
```

### 👉 Time-resampling (binning like light curves)

```python
df["flux_mjy"].resample("10T").mean()
```

Equivalent to binning Vis / X-ray counts in **10-minute exposures**.

---

## 🎨 Matplotlib — Astrophysical Plots

### ⭐ Light Curves

Flux vs Time with error bars:

```python
plt.errorbar(time, flux, yerr=error)
```

Produces `lightcurves.png`

### 📈 Histogram

Check flux distribution — QA / variability hint

Produces `flux_hist.png`

### 🔬 Scatter Plot

Flux vs error — check noise behaviour, photometric quality

Produces `flux_vs_err.png`

---

## 📦 Outputs Generated

| File                  | Description                          |
| --------------------- | ------------------------------------ |
| `observations.csv`    | Sample input data                    |
| `lightcurves.png`     | Flux vs time (optical & radio)       |
| `flux_hist.png`       | Histogram of flux distribution       |
| `flux_vs_err.png`     | Measurement error QA plot            |
| `summary_by_band.csv` | Stats per band (mean, std, min, max) |

---

## 📘 Key Concepts Learned

| Concept      | Astronomy Application                              |
| ------------ | -------------------------------------------------- |
| CSV reading  | telescope logs, beamformed CSVs, photometry tables |
| Time index   | light curve analysis, pulsar timing                |
| Grouping     | multi-band photometry, detector comparison         |
| Resampling   | binning photon counts/time slicing                 |
| NumPy arrays | flux arrays, error propagation                     |
| Matplotlib   | plotting calibrated science products               |

---
