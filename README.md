# 🎬 IMDb Movie Success Prediction & Revenue Estimation  
### Data-Science Term Project — Team 3 (2025)

> Predicting whether a film will be a **box-office hit** and how much it will **earn** using IMDb metadata.  
> Course: *Data Science*, Prof. Eom Gwang-hyeon, Spring 2025. :contentReference[oaicite:0]{index=0}

---
## Quick Start 🚀

> **Prerequisites**  
> * Python 3.9 or newer  
> * Git  
> * Graphviz system binaries (`dot`) – see installation note below  

```bash
# ① Clone the repo
git clone https://github.com/nyewon/DS_Team03.git
cd DS_Team03

# ② Create & activate a virtual environment
#    Windows (PowerShell)
python -m venv .venv
.venv\Scripts\Activate.ps1           # older CMD → .venv\Scripts\activate.bat

#    macOS / Linux
python3 -m venv .venv
source .venv/bin/activate

# ③ Install Python dependencies
pip install --upgrade pip
pip install -r requirements.txt      # scikit-learn, pandas, graphviz (Python wrapper) …

# ④ (Windows only) Ensure Graphviz system package is installed
#    Download & run installer: https://graphviz.gitlab.io/download/
#    Add  C:\Program Files\Graphviz\bin  to your PATH, then reopen the terminal.
dot -V                               # should print version info

# ⑤ Run the entire pipeline (EDA → ML → outputs/)
python run_pipeline.py

# ⑥ Run a single stage (e.g. logistic-regression only)
python run_pipeline.py --stage logistic

# ⑦ Deactivate env when done
deactivate
```

## Table of Contents
1. [Problem Statement](#problem-statement)
2. [Dataset](#dataset)
3. [Repository Layout](#repository-layout)
4. [Pre-processing Pipeline](#pre-processing-pipeline)
5. [Exploratory Data Analysis](#exploratory-data-analysis)
6. [Modeling & Results](#modeling--results)
7. [How to Reproduce](#how-to-reproduce)
8. [Team](#team)

---

## Problem Statement
Accurately forecasting movie success **before release** helps  
* studios optimise investment & marketing,  
* streaming platforms plan acquisitions, and  
* audiences gauge hype.  
We therefore built  
* a **classification model** to flag *hit* vs *not-hit*, and  
* a **regression model** to predict **revenue**. :contentReference[oaicite:1]{index=1}

---

## Dataset
| Source | Kaggle – “IMDb Movies Dataset” |
|--------|--------------------------------|
| Size   | 10 k + films |
| Fields | `names`, `date_x`, `score`, `genre`, `crew`, `budget_x`, `revenue`, `status`, `country`, … |

A cleaned & feature-encoded version (`imdb_movies_processed.csv`) lives in [`data/`](data/).  
Full raw dataset link → <https://www.kaggle.com/datasets/ashpalsingh1525/imdb-movies-dataset>

---

---

## Pre-processing Pipeline
| Step | Details |
|------|---------|
| **Missing / extreme values** | • budget < $10 k or revenue = 0 ⇒ `NaN`  <br>• Drop rows with `NaN` in critical cols |
| **Date → Year** | parse `date_x` → `year` (`YYYY`) and drop original |
| **Numeric scaling** | [`RobustScaler`](https://scikit-learn.org/) to tame outliers (`*_scaled`) |
| **Categoricals** | `genre` → **multi-hot** (MultiLabelBinarizer) <br>`status` → one-hot (`drop_first`) |
| **Output** | `imdb_movies_processed.csv` |

---

## Exploratory Data Analysis
* **Revenue** is *right-skewed*; most films earn < $200 M, a few > $1 B.  
* **Budget ↔ Revenue**: clear positive trend; high-budget films succeed more often.  
* **Top success-rate genres**: *TV Movie*, *Documentary*, *Family* (> 80 %).  
* **Yearly trend**: average revenue climbs steadily in recent decades.  
* See `notebooks/EDA.ipynb` for annotated plots. :contentReference[oaicite:2]{index=2}

---

## Modeling & Results

### 1. Revenue Regression
| Model | Best params | 𝑅² | RMSE |
|-------|-------------|----|-----|
| **Decision Tree** | `max_depth=10`, `min_samples_split=10` | **0.65** | 0.40 |
| Decision Tree (depth 5) |  | 0.63 | 0.40 (better generalisation) |

*Feature importance*: `budget_x_scaled` ≫ `score_scaled` ≫ others.  

### 2. Hit/Not-Hit Classification
* **Logistic Regression** with threshold 0.5–0.6 gave best *F1 ≈ 0.89*.  
* **Decision Tree** (Gini / Entropy) traded higher *Precision* for slightly lower recall.  
* Threshold 0.4 maximises recall (catch every hit), 0.7+ maximises precision.

> `budget_x_scaled` and `score_scaled` dominate all classifiers. :contentReference[oaicite:3]{index=3}

---

## How to Reproduce
```bash
# 1. Clone & set up
git clone https://github.com/nyewon/DS_Team03.git
cd DS_Team03
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. Pre-process data
python src/preprocess.py --input data/imdb_movies_raw.csv --output data/imdb_movies_processed.csv

# 3. Train regression
python src/train_regression.py --config configs/reg_tree.yml

# 4. Train classifier
python src/train_classify.py --config configs/logreg.yml
```
---


## Team

| ID        | Name              | Role                              |
| --------- | ----------------- | --------------------------------- |
| 201935027 | **Kim Jae-hee**   | Regression, slides                |
| 201935147 | **Chu Dong-hyuk** | Classification, presentation      |
| 202131791 | **Han Won-geun**  | Pre-processing, GitHub/Kaggle ops |
| 202234885 | **No Ye-won**     | EDA & visualisation               |

---
