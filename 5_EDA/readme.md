# 🛰️ Exploratory Data Analysis (EDA) for Astronomical Catalogs

### *Example: SDSS DR18 — Stars vs Galaxies vs Quasars*

Exploratory Data Analysis (EDA) is the **first and most critical step** before any ML task in astrophysics — classification, regression, clustering, anomaly search, etc.

Astronomy datasets often include:

* Object classes: **STAR / GALAXY / QSO (quasar)**
* Photometric measurements: **u, g, r, i, z** magnitudes
* Spectral features: **redshift (z)**
* Positional data: **RA, DEC**
* Quality flags & measurement errors

Understanding them **before** machine learning ensures realistic results, avoids biases, and protects against garbage-in garbage-out.

---

## ✅ Step 1 — Load Dataset

```python
import pandas as pd

df = pd.read_csv("SDSS_DR18.csv")
```

📌 SDSS = **Sloan Digital Sky Survey**
DR18 = latest major public release (millions of objects)

---

## ✅ Step 2 — First Look at Data

### 🔭 View first few rows

```python
df.head()
```

**Why?**

* Sanity check file loaded correctly
* See column names: magnitudes, redshift, RA/DEC, class
* Ensure no weird encodings / corruption

---

### 🧠 Dataset Info

```python
df.info()
```

Tells us:

| Property        | Meaning                                   |
| --------------- | ----------------------------------------- |
| Column names    | What features exist (magnitudes, RA, etc) |
| Data types      | float, int, object (important for ML)     |
| Non-null counts | Missing value detection                   |
| Memory usage    | Performance planning                      |

For astronomy, dtype correctness matters
→ `object` instead of `float` can break pipelines.

---

### 📊 Summary Statistics

```python
df.describe()
```

For numeric columns — gives:

* mean flux/magnitude
* standard deviation → variability of sources
* min/max → catch anomalies (negative flux? impossible)
* quartiles → distribution shape

Important for astrophysics:

| Metric           | Meaning                              |
| ---------------- | ------------------------------------ |
| Very large std   | wide magnitude range (stars vs QSOs) |
| Extreme outliers | likely **instrumental artifacts**    |
| Negative values  | instrument or calibration issue      |

---

### 🌌 Object Class Distribution

```python
df["class"].value_counts()
```

Astronomy imbalance example:

| Class        | Count |
| ------------ | ----: |
| GALAXY       |  huge |
| STAR         |  many |
| QSO (quasar) | fewer |

**Why it matters:**
Unbalanced dataset → biased model.
Need balancing / weighting / stratified sampling.

---

### 🕳️ Missing Value Check

```python
df.isna().sum().sort_values(ascending=False).head(20)
```

Missing magnitudes & redshifts are common due to:

* sensor saturation
* faint sources
* fiber assignment limits
* failed spectral extraction

🧠 What to do with missing values?

| Option                   | When                      |
| ------------------------ | ------------------------- |
| Drop rows                | few NaNs, easy win        |
| Impute with median / KNN | science-safe method       |
| Use `-999` sentinel      | for astrophysical surveys |
| Model missing as feature | ML sometimes benefits     |

---

## 📌 Why EDA Matters in Astro-ML?

| Problem                   | Impact                                     |
| ------------------------- | ------------------------------------------ |
| Missing redshift          | model can't learn distance physics         |
| Skewed class distribution | predicts only galaxies                     |
| Dirty magnitudes          | wrong color indices = wrong classification |
| Wrong dtype               | ML crashes / wrong math                    |
| Outliers                  | biases astrophysical scaling laws          |

In astronomy:
**Bad preprocessing > Bad science > Wrong conclusions**

---

## 🔎 Key Astro-Specific EDA Questions

| Check                 | Question                               |
| --------------------- | -------------------------------------- |
| Magnitude ranges      | Are magnitudes realistic (0–25)?       |
| Redshift distribution | Peaks at ~0 for stars, ~2 for quasars? |
| Colors (g-r, u-g)     | Do they separate classes clearly?      |
| Spatial coverage      | Uniform RA/DEC? Survey footprint bias? |
| Photometric errors    | Higher noise = fainter objects?        |

---

## 📈 Typical Plots After This Stage

| Type                      | Purpose                                         |
| ------------------------- | ----------------------------------------------- |
| Histogram (u, g, r, i, z) | brightness distribution                         |
| Scatter (u-g vs g-r)      | color-color diagram → star/galaxy/QSO separator |
| Redshift histogram        | cosmic structure hint                           |
| Class imbalance bar plot  | modeling strategy                               |
| Heatmap                   | correlation between magnitudes                  |

---

## 🧠 Summary

After this EDA step, we understand:

* ✅ Data size & structure
* ✅ Column & data types
* ✅ Distribution of astro objects
* ✅ Missing value patterns
* ✅ Ready for next steps:

### 🚀 Next Steps in Astro-ML Pipeline

| Stage                 | Goal                                    |
| --------------------- | --------------------------------------- |
| Feature engineering   | colors, log-flux, extinction correction |
| Outlier removal       | eliminate detector failures             |
| Train/val/test split  | stratified by class                     |
| Scaling/normalization | magnitudes → colors                     |
| Modeling              | RF, SVM, NN, CNN for spectra            |
| Evaluation            | precision, recall (esp. quasars)        |

---

