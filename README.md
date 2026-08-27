# Intrusion Detection System (IDS) — CIC-CSE-IDS2018

An Anomaly and Intrusion Detection System implementing **Decision Tree** and **K-Nearest Neighbors (KNN)** classifiers on a subset of the **CIC-CSE-IDS2018** network traffic dataset.

---

## 📌 Project Overview

This repository provides an end-to-end Machine Learning pipeline designed for network intrusion detection. The implementation automates data loading, thorough cleaning and normalization, feature selection using ANOVA F-value, exploratory data analysis (EDA), and comparative evaluation of two supervised learning algorithms.

### Target Attack Types Included
* **Botnet** (`03-02-2018.csv`)
* **Infiltration** (`03-01-2018.csv`)
* **Brute Force — Web / XSS / SQL Injection** (`02-23-2018.csv`)
* **FTP-BruteForce & SSH-BruteForce** (`02-14-2018.csv`)

---

## 🛠️ Pipeline Architecture

1. **Data Ingestion & Cleaning:**
   * Removes embedded duplicate header rows caused by packet capture restarts.
   * Deduplicates clashing column names.
   * Drops non-informative metadata (`Dst Port`, `Timestamp`).
   * One-hot encodes the `Protocol` feature.
   * Coerces numeric features, drops `NaN`s, replaces `Infinity` values with column max, and filters out negative values or duplicates.

2. **Feature Scaling & Selection:**
   * Features normalized using `MinMaxScaler` to [0, 1].
   * Top 10 features selected using **ANOVA F-value** (`SelectKBest`).

3. **Model Training & Evaluation:**
   * **Decision Tree Classifier:** Evaluated using Gini impurity.
   * **K-Nearest Neighbors (KNN):** K=5 using Euclidean distance metric.
   * Dataset split into **70% Training / 30% Testing** with class stratification.

---

## 📁 Repository Structure

```text
.
├── data/                    # Folder containing raw CSV files
│   ├── 02-14-2018.csv
│   ├── 02-23-2018.csv
│   ├── 03-01-2018.csv
│   └── 03-02-2018.csv
├── output/                  # Folder generated automatically for saved figures & results
│   ├── fig2_2_feature_scores.png
│   ├── fig3_1_class_distribution.png
│   ├── fig3_2_violin_top_feature.png
│   ├── fig3_3_violin_second_feature.png
│   ├── fig3_4_correlation_heatmap.png
│   ├── fig5_model_comparison.png
│   ├── cm_decision_tree.png
│   ├── cm_knn.png
│   └── results_summary.csv
├── ids_knn_dt.py            # Main pipeline execution script
└── README.md
```

---

## ⚡ Quick Start

### 1. Prerequisites & Dependencies
Ensure Python 3.8+ is installed. Install required packages using:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn
```

### 2. Dataset Setup
Download the required CSV files from Kaggle:
* 🔗 [Kaggle — IDS Intrusion CSV (CIC-CSE-IDS2018)](https://www.kaggle.com/datasets/solarmainframe/ids-intrusion-csv)

Place the following 4 files into the `./data/` directory:
* `02-14-2018.csv`
* `02-23-2018.csv`
* `03-01-2018.csv`
* `03-02-2018.csv`

### 3. Run the Pipeline
Execute the main script:

```bash
python ids_knn_dt.py
```

---

## 📊 Outputs & Generated Plots

Running the script automatically generates and saves figures to the `./output/` directory:

* **Feature Selection:** `fig2_2_feature_scores.png` — Horizontal bar chart of ANOVA F-scores.
* **Class Frequency:** `fig3_1_class_distribution.png` — Distribution percentage per class.
* **Feature Distributions:** `fig3_2_violin_top_feature.png` & `fig3_3_violin_second_feature.png` — Violin plots comparing Benign vs Malicious behavior.
* **Correlation Heatmap:** `fig3_4_correlation_heatmap.png` — Correlation matrix between top features and the label.
* **Confusion Matrices:** `cm_decision_tree.png` and `cm_knn.png` — Detailed confusion matrices per classifier.
* **Performance Comparison:** `fig5_model_comparison.png` & `results_summary.csv` — Comparative metrics overview.

---

## 🚀 Future Enhancements

* Hyperparameter tuning using `GridSearchCV` or `RandomizedSearchCV`.
* Implementing **SMOTE** to handle class imbalance.
* Evaluating deep neural network architectures (DNN/LSTM) on the full dataset.
