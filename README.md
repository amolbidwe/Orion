# Python for Data Science Workshop

> **Goal:** Take participants from absolute Python basics (print statements) to performing Exploratory Data Analysis (EDA), visualization, and classical machine learning on a real dataset (SDSS example).

---

## Table of Contents

1. Python basics (quick refresher)
2. Working with files and the filesystem
3. Python data structures useful for data science
4. Scientific Python ecosystem (NumPy, Pandas, Matplotlib)
6. Loading and inspecting data (EDA)
7. Data cleaning and preprocessing
8. Data visualization — from basic to informative plots
9. Feature engineering and selection
10. Classical Machine Learning workflow
11. Model evaluation and metrics
12. Model selection and hyperparameter tuning
13. Putting it together: end-to-end example (SDSS dataset)
14. Best practices & tips for workshops
15. Exercises and suggested next steps


---

## 1 — Python basics (quick refresher)

```python
# print, variables, types
print("Hello, Data Science!")
name = "Swapnil"
age = 28
pi = 3.14159

# basic types
print(type(name), type(age), type(pi))

# conditionals
if age > 18:
    print("Adult")
else:
    print("Minor")

# loops
for i in range(5):
    print(i)

# functions
def square(x):
    return x*x

print(square(5))

# list comprehensions
squares = [x*x for x in range(10)]
print(squares)
```

> **Note:** This workshop assumes you are comfortable with the short refresher above. If not, spend time on a Python basics tutorial first.

---

## 2 — Working with files and the filesystem

```python
# list files
import os
print(os.listdir('.'))

# read a text file
with open('example.txt', 'r') as f:
    s = f.read()
print(s[:200])
```

---

## 3 — Python data structures useful for DS

* `list`, `tuple`, `set`, `dict`
* `numpy.ndarray` for numeric arrays
* `pandas.DataFrame` and `pandas.Series` for tabular data

Example:

```python
import numpy as np
import pandas as pd
arr = np.arange(12).reshape(3,4)
df = pd.DataFrame(arr, columns=['a','b','c','d'])
print(df.head())
```

---

## 4 — Scientific Python ecosystem

* **NumPy** — fast numeric arrays
* **Pandas** — data frames, read/write CSV, groupby, joins
* **Matplotlib** — core plotting library
* **Seaborn** — statistical plotting on top of Matplotlib
* **scikit-learn** — classical ML algorithms and utilities

Install and quick imports:

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
```

---

## 5 — Loading and inspecting data (EDA)

**Load CSV**

```python
df = pd.read_csv('SDSS_DR18.csv')
```

**Quick checks**

```python
# top rows
df.head()
# summary
df.info()
# numerical description
df.describe()
# class distribution
df['class'].value_counts()
# missing values
df.isna().sum().sort_values(ascending=False).head(20)
```

**Why these checks?** They tell you shape, types, presence of nulls, and class balance — essential before modeling.

---

## 6 — Data cleaning and preprocessing

**Common steps**

1. Remove irrelevant columns (IDs, urls)
2. Handle missing values (drop or impute)
3. Convert datatypes
4. Encode categorical variables (LabelEncoder / OneHot)
5. Scale numeric features using `StandardScaler` or `MinMaxScaler`

**Examples**

```python
# drop
for c in ['objid','specobjid']:
    if c in df.columns:
        df.drop(columns=c, inplace=True)

# fill missing with median
num_cols = df.select_dtypes(include=['int64','float64']).columns
for c in num_cols:
    df[c].fillna(df[c].median(), inplace=True)

# label encode target if needed
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df['target_enc'] = le.fit_transform(df['class'])
```

**Train-test split (stratified)**

```python
X = df.drop(columns=['class','target_enc'])
y = df['target_enc']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
```

---

## 7 — Data visualization

**Quick plotting rules**

* Start with distributions of individual features (histograms)
* Check pairwise relationships (scatter plots) for important features
* Plot correlation heatmap for numeric features
* Visualize class balance (bar / pie chart)

**Examples**

```python
# distribution
plt.hist(df['u'], bins=50)
plt.title('u mag distribution')
plt.show()

# seaborn countplot for classes
sns.countplot(x='class', data=df)
plt.show()

# correlation heatmap (sample if data large)
num = df.select_dtypes(include=['number']).sample(5000, random_state=1)
corr = num.corr()
sns.heatmap(corr, annot=False)
plt.title('Correlation matrix')
plt.show()

# sky coordinates scatter (ra, dec) colored by class
sns.scatterplot(data=df.sample(5000), x='ra', y='dec', hue='class', s=5)
plt.title('Equatorial coordinates by class')
plt.show()
```

> Tip: For very large datasets sample before plotting to keep visuals readable.

---

## 8 — Feature engineering & selection

* Create domain-specific features (ratios, differences)
* Remove features with low variance
* Use correlation to drop highly correlated features
* Consider tree-based importance (RandomForest) for selection

```python
# example: color indices in astronomy (u-g, g-r, r-i, i-z)
df['u_g'] = df['u'] - df['g']
df['g_r'] = df['g'] - df['r']

# drop near-constant columns
from sklearn.feature_selection import VarianceThreshold
sel = VarianceThreshold(threshold=0.0)
sel.fit(df.select_dtypes(include=['number']))
```

---

## 9 — Classical ML workflow (repeatable)

1. Problem formulation (classification/regression)
2. Data cleaning & split
3. Feature scaling/encoding
4. Model selection & training
5. Evaluation on validation/test
6. Tuning & repeat

**Common models to try**

* Logistic Regression (baseline, probabilistic)
* Support Vector Machines / LinearSVC
* Decision Tree
* Random Forest
* K-Nearest Neighbors
* Gradient Boosting (XGBoost/LightGBM if installed)

Example pipeline for Logistic Regression:

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('clf', LogisticRegression(multi_class='multinomial', max_iter=2000))
])
pipe.fit(X_train, y_train)
y_pred = pipe.predict(X_test)
```

---

## 10 — Model evaluation & metrics

**Classification metrics**

* Accuracy (overall)
* Precision, Recall, F1-score (per-class)
* Confusion matrix
* ROC-AUC (for binary or one-vs-rest)

Example:

```python
from sklearn.metrics import classification_report, confusion_matrix
print(classification_report(y_test, y_pred))
cm = confusion_matrix(y_test, y_pred)

# plot
from sklearn.metrics import ConfusionMatrixDisplay
labels = le.classes_ if 'le' in globals() else sorted(df['class'].unique())
ConfusionMatrixDisplay(cm, display_labels=labels).plot()
plt.show()
```

---

## 11 — Model selection & hyperparameter tuning

* Use `GridSearchCV` or `RandomizedSearchCV` with cross-validation
* Build pipelines to avoid data leakage
* Use `StratifiedKFold` for classification

Example GridSearch for LinearSVC:

```python
from sklearn.svm import LinearSVC
from sklearn.model_selection import GridSearchCV, StratifiedKFold

pipe = Pipeline([('scaler', StandardScaler()), ('clf', LinearSVC(max_iter=5000))])
param_grid = {'clf__C': [0.01, 0.1, 1, 10]}
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
grid = GridSearchCV(pipe, param_grid, cv=cv, scoring='accuracy', n_jobs=-1)
grid.fit(X_train, y_train)
print(grid.best_params_)
```

---

## 12 — End-to-end example (SDSS dataset)

This section shows a compact, ready-to-run notebook workflow that follows everything from loading to evaluation.

```python
# 1. imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

# 2. load
df = pd.read_csv('SDSS_DR18.csv')
print(df.shape)

# 3. quick EDA
print(df['class'].value_counts())
print(df.describe())

# 4. minimal cleaning
for c in ['objid','specobjid']:
    if c in df.columns:
        df.drop(columns=c, inplace=True)

# fill na with median for numeric
num_cols = df.select_dtypes(include=['int64','float64']).columns
for c in num_cols:
    df[c].fillna(df[c].median(), inplace=True)

# 5. target encode
le = LabelEncoder()
df['target'] = le.fit_transform(df['class'])

# 6. features & split
X = df.select_dtypes(include=['number']).drop(columns=['target'], errors='ignore')
y = df['target']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 7. pipeline & grid search (logistic)
pipe = Pipeline([('scaler', StandardScaler()), ('clf', LogisticRegression(max_iter=2000))])
param_grid = {'clf__C': [0.01, 0.1, 1, 10]}
cv = StratifiedKFold(n_splits=4, shuffle=True, random_state=1)
grid = GridSearchCV(pipe, param_grid, cv=cv, scoring='accuracy', n_jobs=-1)
grid.fit(X_train, y_train)
print('Best params', grid.best_params_)

# 8. evaluate
best = grid.best_estimator_
y_pred = best.predict(X_test)
print(classification_report(y_test, y_pred, target_names=le.classes_))
cm = confusion_matrix(y_test, y_pred)
ConfusionMatrixDisplay(cm, display_labels=le.classes_).plot()
plt.show()

# 9. RandomForest quick baseline
rf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
y_rf = rf.predict(X_test)
print('RF acc', (y_rf==y_test).mean())
```

---

## 13 — Best practices & tips

* Always check class balance and missing values before training.
* Use stratified splits for classification problems.
* Use Pipelines to guarantee safe preprocessing during cross-validation.
* For large datasets, prefer `LinearSVC`, `SGDClassifier`, or subsampling for prototyping.
* Set `random_state` everywhere for reproducible results.
* Log experiments (MLflow or simple CSV) when you run many experiments.

---

## 14 — Exercises and next steps

1. Re-run the notebook but intentionally drop one band (e.g. `u`) — how does model performance change?
2. Try SMOTE to oversample minority classes and compare F1-scores.
3. Compare `LogisticRegression`, `LinearSVC`, and `RandomForest` in a small benchmark table (accuracy + macro F1).
4. Try feature selection using `SelectKBest` and note how performance changes.
5. Create an interactive plot with `plotly` showing RA/Dec colored by predicted class.

---

## Appendix: Useful snippets

* Save model with `joblib`

```python
import joblib
joblib.dump(best, 'best_model.joblib')
model = joblib.load('best_model.joblib')
```

* Save cleaned dataset

```python
df.to_csv('SDSS_cleaned.csv', index=False)
```

---

If you want, I can:

* convert this into a downloadable `.md` file or GitHub-ready README,
* produce a full, runnable `.ipynb` notebook version,
* or tailor the workshop to a specific audience (beginners vs advanced) with exercises and solutions.
