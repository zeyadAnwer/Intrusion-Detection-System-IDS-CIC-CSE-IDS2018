"""
=============================================================================
  Intrusion Detection System (IDS) — CIC-CSE-IDS2018 Dataset
  Algorithms: Decision Tree Classifier | K-Nearest Neighbors (KNN)
  Based on: MTH 410 Data Mining for Cybersecurity — Group 23 Methodology
=============================================================================

DATASET:
  Download from: https://www.kaggle.com/datasets/solarmainframe/ids-intrusion-csv
  Files used (as in the paper):
    - 03-02-2018.csv   (Bot)
    - 03-01-2018.csv   (Infiltration)
    - 02-23-2018.csv   (Brute Force -XSS, Web, SQL Injection)
    - 02-14-2018.csv   (FTP-BruteForce, SSH-BruteForce)

USAGE:
  1. Place the CSV files in a folder (e.g., ./data/)
  2. Update DATA_DIR below if needed
  3. pip install pandas numpy scikit-learn matplotlib seaborn
  4. python ids_knn_dt.py
=============================================================================
"""

import os
import time
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, classification_report, confusion_matrix
)

# =============================================================================
# CONFIGURATION
# =============================================================================

DATA_DIR       = "./data"      # Folder containing the CSV files
OUTPUT_DIR     = "./output"    # Folder for saved plots & reports
TOP_K_FEATURES = 10            # Number of features to select (same as paper)
TEST_SIZE      = 0.30          # 70/30 train-test split
RANDOM_STATE   = 42
KNN_K          = 5             # Number of neighbors for KNN

# Files to load (same subset as the paper)
FILES = [
    "03-02-2018.csv",
    "03-01-2018.csv",
    "02-23-2018.csv",
    "02-14-2018.csv",
]

os.makedirs(OUTPUT_DIR, exist_ok=True)


# =============================================================================
# HELPER — Rename duplicate column names so every column is unique
# =============================================================================

def deduplicate_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Rename duplicate column names by appending a counter suffix.
    e.g. two columns both named 'Fwd Header Length' become
         'Fwd Header Length' and 'Fwd Header Length_1'.
    This prevents 'cannot reindex on an axis with duplicate labels' and
    'Columns must be same length as key' errors.
    """
    seen     = {}
    new_cols = []
    for col in df.columns:
        if col not in seen:
            seen[col] = 0
            new_cols.append(col)
        else:
            seen[col] += 1
            new_cols.append(f"{col}_{seen[col]}")
    df.columns = new_cols
    return df


# =============================================================================
# SECTION 1 — DATA LOADING
# =============================================================================

def load_data(data_dir: str, files: list) -> pd.DataFrame:
    """
    Load and concatenate the selected CSV files.

    Known issues handled:
      (a) Embedded duplicate header rows mid-file — the CIC-IDS2018 CSVs
          repeat the header row whenever the capture tool restarted.
      (b) Duplicate column names — some files have two columns with the
          same name (e.g. 'Fwd Header Length'), which breaks boolean indexing.
    """
    print("\n" + "="*60)
    print("SECTION 1 — DATA LOADING")
    print("="*60)

    dfs = []
    for f in files:
        path = os.path.join(data_dir, f)
        if not os.path.exists(path):
            print(f"  [WARNING] File not found, skipping: {path}")
            continue

        # Read raw CSV; strip whitespace from column names
        df = pd.read_csv(path, low_memory=False)
        df.columns = df.columns.str.strip()

        # -- FIX A: Remove embedded header rows --
        # If the value in column 0 equals the column-0 name, it is a
        # repeated header row.  We detect this BEFORE deduplication so
        # the column name is still its original string.
        first_col   = df.columns[0]
        header_mask = df[first_col].astype(str).str.strip() == str(first_col).strip()
        n_removed   = int(header_mask.sum())
        if n_removed:
            df = df[~header_mask].reset_index(drop=True)
            print(f"  [{f}] Removed {n_removed:,} embedded header row(s)")

        # -- FIX B: Deduplicate column names --
        df = deduplicate_columns(df)

        print(f"  Loaded {f}: {df.shape[0]:,} rows x {df.shape[1]} cols")
        dfs.append(df)

    if not dfs:
        raise FileNotFoundError(
            f"No CSV files found in '{data_dir}'.\n"
            "Please download the dataset from:\n"
            "  https://www.kaggle.com/datasets/solarmainframe/ids-intrusion-csv\n"
            "and place the required files in that folder."
        )

    # Concatenate; deduplicate columns one final time after concat
    combined = pd.concat(dfs, ignore_index=True)
    combined = deduplicate_columns(combined)

    print(f"\n  Combined dataset: {combined.shape[0]:,} rows x {combined.shape[1]} cols")
    return combined


# =============================================================================
# SECTION 2 — DATA PREPROCESSING
# =============================================================================

def preprocess(df: pd.DataFrame) -> tuple:
    """
    Follows the preprocessing pipeline from the paper (Group 23):
      1.  Drop metadata columns (Dst Port, Timestamp)
      2.  One-hot encode 'Protocol'
      3.  Re-deduplicate columns (get_dummies can reintroduce duplicates)
      4.  Coerce all feature columns to numeric, column by column
      5.  Drop NaN rows
      6.  Replace Infinity values with column max
      7.  Drop rows with negative values
      8.  Drop duplicate rows
      9.  Encode target labels with LabelEncoder
      10. Normalize features with MinMaxScaler [0, 1]
      11. Feature selection with SelectKBest (f_classif, top K)
    """
    print("\n" + "="*60)
    print("SECTION 2 — DATA PREPROCESSING")
    print("="*60)
    print(f"  Starting shape: {df.shape}")

    # --- Step 1: Drop metadata columns ---
    drop_cols = [c for c in ["Dst Port", "Timestamp"] if c in df.columns]
    df = df.drop(columns=drop_cols)
    print(f"  Dropped metadata columns: {drop_cols}")

    # --- Step 2: One-hot encode 'Protocol' ---
    if "Protocol" in df.columns:
        df = pd.get_dummies(df, columns=["Protocol"], drop_first=False)
        print("  One-hot encoded 'Protocol' column")

    # --- Step 3: Re-deduplicate after get_dummies ---
    # get_dummies can create new columns whose names clash with existing ones
    df = deduplicate_columns(df)
    print(f"  Column count after one-hot encoding: {df.shape[1]}")

    # Identify the label column
    label_col = "Label"
    if label_col not in df.columns:
        raise ValueError(
            f"Target column '{label_col}' not found. "
            "Check your CSV files — the column may be named differently."
        )

    # Separate feature column names from label
    feature_cols_all = [c for c in df.columns if c != label_col]

    # --- Step 4: Coerce feature columns to numeric, ONE BY ONE ---
    # Using df[list] = df[list].apply(...) fails when the resulting DataFrame
    # has a different number of columns than the key list (happens when column
    # names are still not 100% unique after the dummy step).
    # Assigning column-by-column is always safe.
    n_coerced = 0
    for col in feature_cols_all:
        original = df[col]
        converted = pd.to_numeric(original, errors="coerce")
        if converted.isna().sum() > original.isna().sum():
            n_coerced += 1
        df[col] = converted
    print(f"  Coerced {len(feature_cols_all)} feature columns to numeric "
          f"({n_coerced} had non-numeric values replaced with NaN)")

    # --- Step 5: Drop NaN rows ---
    before = len(df)
    df = df.dropna()
    print(f"  Dropped NaN rows: {before - len(df):,} rows removed")

    # --- Step 6: Replace Infinity values with column max ---
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    feature_cols = [c for c in numeric_cols if c != label_col]

    inf_replaced = 0
    for col in feature_cols:
        inf_mask = np.isinf(df[col])
        if inf_mask.any():
            col_max = df.loc[~inf_mask, col].max()
            df.loc[inf_mask, col] = col_max
            inf_replaced += int(inf_mask.sum())
    print(f"  Replaced {inf_replaced:,} infinity values with column max")

    # --- Step 7: Drop rows with negative values ---
    before   = len(df)
    neg_mask = (df[feature_cols] < 0).any(axis=1)
    df       = df[~neg_mask]
    print(f"  Dropped rows with negative values: {before - len(df):,} rows removed")

    # --- Step 8: Drop duplicate rows ---
    before = len(df)
    df     = df.drop_duplicates()
    print(f"  Dropped duplicate rows: {before - len(df):,} rows removed")
    print(f"  Shape after cleaning: {df.shape}")

    # --- Step 9: Encode target labels ---
    le       = LabelEncoder()
    df[label_col] = le.fit_transform(df[label_col].astype(str))
    label_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
    print(f"\n  Label encoding: {label_mapping}")

    # --- Step 10: Normalize features ---
    feature_cols = [c for c in df.columns if c != label_col]
    scaler       = MinMaxScaler()
    # Assign column-by-column to avoid length-mismatch errors
    scaled = scaler.fit_transform(df[feature_cols])
    for i, col in enumerate(feature_cols):
        df[col] = scaled[:, i]
    print(f"  Normalized {len(feature_cols)} features to [0, 1]")

    # --- Step 11: Feature selection ---
    X = df[feature_cols]
    y = df[label_col]

    k        = min(TOP_K_FEATURES, len(feature_cols))
    selector = SelectKBest(score_func=f_classif, k=k)
    selector.fit(X, y)

    selected_mask     = selector.get_support()
    selected_features = X.columns[selected_mask].tolist()
    feature_scores    = pd.Series(
        selector.scores_[selected_mask], index=selected_features
    ).sort_values(ascending=False)

    print(f"\n  Top {k} features selected (ANOVA F-value):")
    for feat, score in feature_scores.items():
        print(f"    {feat:<40} {score:.2e}")

    X_selected = X[selected_features]
    return X_selected, y, le, feature_scores


# =============================================================================
# SECTION 3 — EXPLORATORY DATA ANALYSIS (EDA)
# =============================================================================

def run_eda(X: pd.DataFrame, y: pd.Series, le: LabelEncoder,
            feature_scores: pd.Series):
    """Generate EDA plots matching the paper's figures."""
    print("\n" + "="*60)
    print("SECTION 3 — EXPLORATORY DATA ANALYSIS (EDA)")
    print("="*60)

    class_names  = le.classes_
    class_counts = y.value_counts().sort_index()
    class_pcts   = class_counts / len(y) * 100

    # ---- Figure 3.1: Class Distribution Histogram ----
    fig, ax = plt.subplots(figsize=(12, 5))
    labels_str = [class_names[i] for i in class_counts.index]
    bars = ax.bar(
        labels_str, class_pcts.values,
        color="#4C72B0", edgecolor="white", linewidth=0.5
    )
    for bar, pct in zip(bars, class_pcts.values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.3,
            f"{pct:.5f}", ha="center", va="bottom", fontsize=8
        )
    ax.set_xlabel("Label")
    ax.set_ylabel("percent")
    ax.set_title(
        "Figure 3.1 — Histogram showing the frequency (%) of the classes in the dataset"
    )
    ax.tick_params(axis="x", rotation=30)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "fig3_1_class_distribution.png"), dpi=150)
    plt.close()
    print("  Saved: fig3_1_class_distribution.png")

    # ---- Figure 3.2: Violin plot — top feature vs Benign / Malicious ----
    top_feat = feature_scores.index[0]
    y_binary = (y > 0).astype(int)
    plot_df  = pd.DataFrame({
        top_feat: X[top_feat].values,
        "Label":  y_binary.map({0: "Benign", 1: "Malicious"}).values
    })
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.violinplot(
        data=plot_df, x=top_feat, y="Label",
        palette=["#4472C4", "#ED7D31"], ax=ax
    )
    ax.set_title(
        f"Figure 3.2 — Violin plot showing the numerical distributions\n"
        f"of '{top_feat}' (Benign vs Malicious)"
    )
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "fig3_2_violin_top_feature.png"), dpi=150)
    plt.close()
    print("  Saved: fig3_2_violin_top_feature.png")

    # ---- Figure 3.3: Violin plot — second feature ----
    if len(feature_scores) > 1:
        second_feat = feature_scores.index[1]
        plot_df2    = pd.DataFrame({
            second_feat: X[second_feat].values,
            "Label":     y_binary.map({0: "Benign", 1: "Malicious"}).values
        })
        fig, ax = plt.subplots(figsize=(7, 5))
        sns.violinplot(
            data=plot_df2, x=second_feat, y="Label",
            palette=["#4472C4", "#ED7D31"], ax=ax
        )
        ax.set_title(
            f"Figure 3.3 — Violin plot showing the numerical distributions\n"
            f"of '{second_feat}' (Benign vs Malicious)"
        )
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, "fig3_3_violin_second_feature.png"), dpi=150)
        plt.close()
        print("  Saved: fig3_3_violin_second_feature.png")

    # ---- Figure 3.4: Correlation Heatmap ----
    corr_df            = X.copy()
    corr_df["Label"]   = y.values
    corr_matrix        = corr_df.corr()

    fig, ax = plt.subplots(figsize=(13, 11))
    sns.heatmap(
        corr_matrix, annot=True, fmt=".2f", cmap="RdYlBu_r",
        center=0, linewidths=0.5, ax=ax, annot_kws={"size": 7}
    )
    ax.set_title("Figure 3.4 — Correlation Heatmap with the selected features")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "fig3_4_correlation_heatmap.png"), dpi=150)
    plt.close()
    print("  Saved: fig3_4_correlation_heatmap.png")

    # ---- Figure 2.2: Feature Selection Scores ----
    fig, ax = plt.subplots(figsize=(9, 5))
    feature_scores.sort_values().plot(kind="barh", ax=ax, color="#4C72B0")
    ax.set_xlabel("ANOVA F-Score")
    ax.set_title(
        "Figure 2.2 — Table showing the selected features scores in descending form"
    )
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "fig2_2_feature_scores.png"), dpi=150)
    plt.close()
    print("  Saved: fig2_2_feature_scores.png")

    print("  EDA complete.")


# =============================================================================
# SECTION 4 — MODEL TRAINING & EVALUATION
# =============================================================================

def plot_confusion_matrix(cm, class_names, title, filename):
    """Plot and save a labeled confusion matrix."""
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=class_names, yticklabels=class_names,
        ax=ax, linewidths=0.5
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title)
    plt.xticks(rotation=35, ha="right", fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, filename), dpi=150)
    plt.close()
    print(f"  Saved: {filename}")


def evaluate_model(model, X_test, y_test, class_names, model_name, cm_filename):
    """Run prediction, compute all metrics, print report, save confusion matrix."""
    y_pred = model.predict(X_test)

    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    rec  = recall_score(y_test,  y_pred,   average="weighted", zero_division=0)
    f1   = f1_score(y_test,      y_pred,   average="weighted", zero_division=0)

    print(f"\n  Accuracy : {acc:.4f}")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall   : {rec:.4f}")
    print(f"  F1-score : {f1:.4f}")
    print(f"\n  Per-class report:")
    print(classification_report(
        y_test, y_pred, target_names=class_names, zero_division=0
    ))

    cm = confusion_matrix(y_test, y_pred)
    plot_confusion_matrix(
        cm, class_names,
        title=f"Confusion Matrix — {model_name}",
        filename=cm_filename
    )

    return {"Accuracy": acc, "Precision": prec, "Recall": rec, "F1-score": f1}


# --------------------------------------------------------------------------
# 4.1 — Decision Tree Classifier
# --------------------------------------------------------------------------

def train_decision_tree(X_train, X_test, y_train, y_test, class_names):
    """
    Decision Trees is a supervised machine learning algorithm, useful for both
    Classification and Regression problems. It predicts the output label by
    learning simple decision rules inferred from the input features — like a
    series of If-Then-Else tests that lead to the final output.

    Criterion: Gini Impurity — measures the probability of a random instance
    being misclassified when chosen randomly.
    Lower Gini index = purer split = lower likelihood of misclassification.
    """
    print("\n" + "="*60)
    print("SECTION 4.1 — DECISION TREE CLASSIFIER")
    print("="*60)
    print("  Algorithm : Decision Tree (supervised, classification)")
    print("  Criterion : Gini Impurity")
    print("  Measures  : Quality of each split (lower = purer = better)")

    dt = DecisionTreeClassifier(criterion="gini", random_state=RANDOM_STATE)

    start      = time.time()
    dt.fit(X_train, y_train)
    train_time = time.time() - start
    print(f"  Training time: {train_time:.2f} seconds")

    metrics = evaluate_model(
        dt, X_test, y_test, class_names,
        model_name="Decision Tree",
        cm_filename="cm_decision_tree.png"
    )
    metrics["Execution Time"] = f"{train_time:.2f}s"
    return dt, metrics


# --------------------------------------------------------------------------
# 4.2 — K-Nearest Neighbors (KNN) Classifier
# --------------------------------------------------------------------------

def train_knn(X_train, X_test, y_train, y_test, class_names):
    """
    KNN is a non-parametric, instance-based supervised learning algorithm.
    It classifies a new data point by finding the K closest training samples
    in feature space (using Euclidean distance) and assigning the majority
    class label among those neighbors.

    Advantages for IDS:
      - No explicit model training — adapts naturally to new traffic patterns.
      - Simple, intuitive decision boundary.
    Disadvantages:
      - Prediction time is O(n * d) per query — slow on large test sets.
      - Sensitive to feature scale, which is why MinMaxScaler is applied first.

    We use n_jobs=-1 to parallelize distance computation across all CPU cores.
    """
    print("\n" + "="*60)
    print("SECTION 4.2 — K-NEAREST NEIGHBORS (KNN) CLASSIFIER")
    print("="*60)
    print(f"  Algorithm       : KNN (instance-based, non-parametric)")
    print(f"  K (neighbors)   : {KNN_K}")
    print(f"  Distance metric : Euclidean (Minkowski p=2)")
    print(f"  Parallel jobs   : -1 (all available CPU cores)")
    print(f"  NOTE: Prediction on a large test set may take a few minutes.")

    knn = KNeighborsClassifier(
        n_neighbors=KNN_K,
        metric="minkowski",
        p=2,
        n_jobs=-1
    )

    start      = time.time()
    knn.fit(X_train, y_train)
    train_time = time.time() - start
    print(f"  Training time (index build): {train_time:.2f} seconds")

    start_pred = time.time()
    metrics    = evaluate_model(
        knn, X_test, y_test, class_names,
        model_name=f"KNN (K={KNN_K})",
        cm_filename="cm_knn.png"
    )
    pred_time = time.time() - start_pred
    metrics["Execution Time"] = f"train={train_time:.2f}s / predict={pred_time:.2f}s"
    return knn, metrics


# =============================================================================
# SECTION 5 — CONCLUSION & FINAL SUMMARY
# =============================================================================

def final_summary(results: dict):
    """Print final comparison table and save bar chart — mirrors Table 5.1 in the paper."""
    print("\n" + "="*60)
    print("SECTION 5 — FINAL RESULTS SUMMARY  (Table 5.1)")
    print("="*60)

    summary_df = pd.DataFrame(results).T
    print(f"\n{summary_df.to_string()}\n")

    # ---- Bar chart: Model comparison ----
    metrics_to_plot = ["Accuracy", "Precision", "Recall", "F1-score"]
    plot_df         = summary_df[metrics_to_plot].astype(float)

    fig, ax = plt.subplots(figsize=(9, 5))
    x       = np.arange(len(metrics_to_plot))
    width   = 0.35
    colors  = ["#4472C4", "#ED7D31"]

    for i, (model_name, row) in enumerate(plot_df.iterrows()):
        bars = ax.bar(
            x + i * width, row.values, width,
            label=model_name, color=colors[i], edgecolor="white"
        )
        ax.bar_label(bars, fmt="%.3f", fontsize=8, padding=2)

    ax.set_xticks(x + width / 2)
    ax.set_xticklabels(metrics_to_plot)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Score")
    ax.set_title("Table 5.1 — Final Results: Decision Tree vs KNN")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "fig5_model_comparison.png"), dpi=150)
    plt.close()
    print("  Saved: fig5_model_comparison.png")

    # ---- Save CSV ----
    summary_path = os.path.join(OUTPUT_DIR, "results_summary.csv")
    summary_df.to_csv(summary_path)
    print(f"  Saved: results_summary.csv")

    # ---- Conclusion text ----
    best_f1    = plot_df["F1-score"].astype(float)
    best_model = best_f1.idxmax()

    print("\n" + "="*60)
    print("CONCLUSION")
    print("="*60)
    print(f"""
  Through the utilization of the CSE-CIC-IDS2018 dataset, we built an
  anomaly / intrusion detection system using two supervised classifiers:

    1. Decision Tree  — Gini criterion, fast training (~seconds),
                        highly interpretable split rules.
    2. KNN (K={KNN_K})        — Instance-based learner, Euclidean distance,
                        no explicit model assumptions, slower at
                        prediction time on large datasets.

  Preprocessing pipeline (same as Group 23):
    - Dropped metadata columns  (Dst Port, Timestamp)
    - One-hot encoded Protocol column
    - Removed NaN rows, replaced Infinity with column max
    - Dropped rows with negative values and duplicate rows
    - LabelEncoded target labels
    - MinMaxScaler normalization to [0, 1]
    - Top {TOP_K_FEATURES} features selected via ANOVA F-value (SelectKBest)
    - Train / Test split: 70% / 30%

  Best performing model by weighted F1-score: {best_model}

  Future improvements:
    - Hyperparameter tuning (GridSearchCV / RandomizedSearchCV)
    - Cross-validation (k-fold) for more robust estimates
    - Addressing class imbalance with SMOTE or class_weight parameter
    - Testing on the full 400 GB dataset covering all attack types
    - Comparing with Deep Neural Networks (DNN)
""")


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("\n" + "#"*60)
    print("  IDS — CIC-CSE-IDS2018  |  Decision Tree + KNN")
    print("#"*60)

    # 1. Load data
    df = load_data(DATA_DIR, FILES)

    # 2. Preprocess
    X, y, le, feature_scores = preprocess(df)
    class_names = le.classes_

    # 3. EDA
    run_eda(X, y, le, feature_scores)

    # 4. Train/test split (70% train — 30% test, same as paper)
    print("\n  Splitting data: 70% train / 30% test")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y          # preserve class proportions in both splits
    )
    print(f"  Train: {X_train.shape[0]:,} samples | Test: {X_test.shape[0]:,} samples")

    # 5. Train & evaluate both models
    _, dt_metrics  = train_decision_tree(X_train, X_test, y_train, y_test, class_names)
    _, knn_metrics = train_knn(X_train, X_test, y_train, y_test, class_names)

    # 6. Final summary
    all_results = {
        "Decision Tree" : dt_metrics,
        f"KNN (K={KNN_K})" : knn_metrics,
    }
    final_summary(all_results)

    print(f"\n  All outputs saved to: {os.path.abspath(OUTPUT_DIR)}/")
    print("  Done.\n")


if __name__ == "__main__":
    main()