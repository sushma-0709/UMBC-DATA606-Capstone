#!/usr/bin/env python3
"""
Wallet Classification - Balanced Approach
This script uses improved strategies for handling imbalanced wallet data:
- KMeans with k=10 clusters (better granularity)
- class_weight='balanced' for all classifiers (handles imbalance)
- Detailed per-cluster analysis
"""

import os
import tomllib
import numpy as np
import pandas as pd
import time
from dotenv import load_dotenv
from snowflake.snowpark import Session
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    precision_score, recall_score, f1_score,
    calinski_harabasz_score, balanced_accuracy_score
)

# Load environment variables
load_dotenv()

print("=" * 80)
print("WALLET CLASSIFICATION - BALANCED APPROACH")
print("=" * 80)

# ============================================================================
# 1. CONNECT TO SNOWFLAKE
# ============================================================================
print("\n1. Connecting to Snowflake...")

with open(os.environ["SNOWFLAKE_TOML"], "rb") as f:
    config = tomllib.load(f)

wallet_classification_config = config["connections"]["wallet_classification"]
conn_params = {k: os.path.expandvars(v) for k, v in wallet_classification_config.items()}

session = Session.builder.configs(conn_params).create()
print(f"   ✓ Connected - Role: {session.sql('SELECT CURRENT_ROLE()').collect()[0][0]}")

session.use_schema("ANALYTICS")

# ============================================================================
# 2. LOAD AND CLEAN DATA
# ============================================================================
print("\n2. Loading data from Snowflake...")

df = session.table("WALLET_CLASSIFICATION")
df_wallet = df.to_pandas()
df_wallet_clean = df_wallet.dropna().copy()  # Explicit copy to avoid SettingWithCopyWarning

print(f"   ✓ Loaded {len(df_wallet_clean):,} wallets")

# ============================================================================
# 3. PREPARE FEATURES
# ============================================================================
print("\n3. Preparing features...")

features = [
    "ACTIVE_DAYS",
    "TOTAL_TX_COUNT",
    "TOTAL_OUTGOING_TX",
    "LIFETIME_SENT_LTC",
    "LIFETIME_RECEIVED_LTC",
    "CURRENT_BALANCE_LTC",
    "AVG_SENT_PER_DAY",
    "AVG_RECEIVED_PER_DAY"
]

X = df_wallet_clean[features].copy()
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print(f"   ✓ Feature matrix: {X_scaled.shape}")

# ============================================================================
# 4. KMEANS CLUSTERING WITH k=10 (IMPROVED)
# ============================================================================
print("\n4. Running KMeans Clustering with k=10...")

# Use 10 clusters for better granularity
kmeans = KMeans(n_clusters=10, random_state=42, n_init=10)
df_wallet_clean['cluster'] = kmeans.fit_predict(X_scaled)

centers = pd.DataFrame(
    scaler.inverse_transform(kmeans.cluster_centers_),
    columns=features
)

calinski = calinski_harabasz_score(X_scaled, df_wallet_clean["cluster"])
print(f"   ✓ Calinski-Harabasz Score: {calinski:.3f}")
print(f"   ✓ Number of clusters: 10")

# ============================================================================
# 5. ANALYZE CLUSTER CHARACTERISTICS
# ============================================================================
print("\n5. Analyzing cluster characteristics...")

cluster_summary = []
for i in range(10):
    cluster_data = df_wallet_clean[df_wallet_clean['cluster'] == i]
    summary = {
        'Cluster': i,
        'Count': len(cluster_data),
        'Pct': 100 * len(cluster_data) / len(df_wallet_clean),
        'Avg_Balance': cluster_data['CURRENT_BALANCE_LTC'].mean(),
        'Avg_TX': cluster_data['TOTAL_TX_COUNT'].mean(),
        'Avg_Days': cluster_data['ACTIVE_DAYS'].mean(),
        'Avg_Received': cluster_data['LIFETIME_RECEIVED_LTC'].mean()
    }
    cluster_summary.append(summary)

cluster_df = pd.DataFrame(cluster_summary)
cluster_df = cluster_df.sort_values('Avg_Balance', ascending=False)

print("\n   Cluster Summary (sorted by avg balance):")
print(cluster_df.to_string(index=False))

# ============================================================================
# 6. WALLET-LEVEL CLASSIFICATION (IMPROVED APPROACH)
# ============================================================================
print("\n6. Classifying wallets into categories (wallet-level approach)...")

# Strategy: Classify each wallet individually based on its characteristics
# This provides much better balance than cluster-based classification

# Define percentile thresholds for each category
WHALE_BALANCE_PERCENTILE = 0.99        # Top 1% by balance
WHALE_MIN_TX = 10                       # Must have some activity
EXCHANGE_TX_PERCENTILE = 0.95          # Top 5% by transaction volume
EXCHANGE_MIN_BALANCE = 0.1             # Must have some balance
EXCHANGE_RECEIVED_PERCENTILE = 0.95    # OR top 5% by received amount

# Calculate thresholds
whale_balance_threshold = df_wallet_clean['CURRENT_BALANCE_LTC'].quantile(WHALE_BALANCE_PERCENTILE)
exchange_tx_threshold = df_wallet_clean['TOTAL_TX_COUNT'].quantile(EXCHANGE_TX_PERCENTILE)
exchange_received_threshold = df_wallet_clean['LIFETIME_RECEIVED_LTC'].quantile(EXCHANGE_RECEIVED_PERCENTILE)

print(f"\n   Wallet-level classification thresholds:")
print(f"     Whale balance (99th percentile): {whale_balance_threshold:.4f} LTC")
print(f"     Whale minimum TX count: {WHALE_MIN_TX}")
print(f"     Exchange TX volume (95th percentile): {exchange_tx_threshold:.0f} transactions")
print(f"     Exchange received (95th percentile): {exchange_received_threshold:.4f} LTC")
print(f"     Exchange minimum balance: {EXCHANGE_MIN_BALANCE} LTC")

# Classify each wallet based on its individual characteristics
def classify_wallet(row):
    balance = row['CURRENT_BALANCE_LTC']
    tx_count = row['TOTAL_TX_COUNT']
    received = row['LIFETIME_RECEIVED_LTC']

    # Whale: Very high balance AND some activity (not dormant)
    if balance >= whale_balance_threshold and tx_count >= WHALE_MIN_TX:
        return 'Whale'

    # Exchange/Merchant: Very high transaction activity OR very high received amount
    # Exchanges have lots of transactions and receive large amounts
    elif (tx_count >= exchange_tx_threshold or received >= exchange_received_threshold) and balance >= EXCHANGE_MIN_BALANCE:
        return 'Exchange/Merchant'

    # Individual: Everyone else (normal users)
    else:
        return 'Individual'

df_wallet_clean['wallet_category'] = df_wallet_clean.apply(classify_wallet, axis=1)

print("\n   Final Category Distribution:")
category_counts = df_wallet_clean['wallet_category'].value_counts().sort_index()
for category, count in category_counts.items():
    pct = 100 * count / len(df_wallet_clean)
    print(f"     {category:20s}: {count:,} ({pct:.2f}%)")

# Show cluster distribution across categories (for analysis)
print("\n   Category distribution across clusters:")
category_cluster_dist = pd.crosstab(
    df_wallet_clean['wallet_category'],
    df_wallet_clean['cluster'],
    margins=True
)
print(category_cluster_dist.to_string())

# ============================================================================
# 7. TRAIN/TEST SPLIT
# ============================================================================
print("\n7. Splitting data for classification...")

y = df_wallet_clean['cluster'].values

# Check if stratification is possible (all classes need at least 2 samples)
unique, counts = np.unique(y, return_counts=True)
min_samples = counts.min()

if min_samples >= 2:
    print(f"   ✓ Using stratified split (min samples per class: {min_samples})")
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )
else:
    print(f"   ⚠ Cannot stratify - some clusters have only {min_samples} sample(s)")
    print(f"   ✓ Using random split instead")
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )

print(f"   ✓ Training set: {len(X_train):,}")
print(f"   ✓ Test set: {len(X_test):,}")

# ============================================================================
# 8. CLASSIFICATION MODELS WITH BALANCED WEIGHTS
# ============================================================================
print("\n" + "=" * 80)
print("CLASSIFICATION MODELS (with class_weight='balanced')")
print("=" * 80)

# Random Forest with balanced weights
print("\n8a. Random Forest Classifier (balanced)...")
start_time = time.time()
rf_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    class_weight='balanced',  # Handle imbalance
    random_state=42,
    n_jobs=-1,
    verbose=0
)
rf_model.fit(X_train, y_train)
rf_time = time.time() - start_time
y_pred_rf = rf_model.predict(X_test)
accuracy_rf = accuracy_score(y_test, y_pred_rf)
balanced_acc_rf = balanced_accuracy_score(y_test, y_pred_rf)
print(f"   ✓ Training time: {rf_time:.2f}s")
print(f"   ✓ Accuracy: {accuracy_rf:.4f}")
print(f"   ✓ Balanced Accuracy: {balanced_acc_rf:.4f}")

# LightGBM with balanced weights
print("\n8b. LightGBM Classifier (balanced)...")
try:
    import lightgbm as lgb
except ImportError:
    print("   ⚠ lightgbm not found. Installing...")
    import sys
    import subprocess
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "lightgbm"])
        import lightgbm as lgb
        print("   ✓ lightgbm installed successfully")
    except Exception as e:
        print(f"   ✗ Failed to install lightgbm: {e}")
        raise

start_time = time.time()
lgbm_model = lgb.LGBMClassifier(
    n_estimators=100,
    num_leaves=31,
    learning_rate=0.1,
    class_weight='balanced',  # Handle imbalance
    random_state=42,
    verbose=-1,
    n_jobs=-1
)
lgbm_model.fit(X_train, y_train)
lgbm_time = time.time() - start_time
y_pred_lgbm = lgbm_model.predict(X_test)
accuracy_lgbm = accuracy_score(y_test, y_pred_lgbm)
balanced_acc_lgbm = balanced_accuracy_score(y_test, y_pred_lgbm)
print(f"   ✓ Training time: {lgbm_time:.2f}s")
print(f"   ✓ Accuracy: {accuracy_lgbm:.4f}")
print(f"   ✓ Balanced Accuracy: {balanced_acc_lgbm:.4f}")

# Logistic Regression with balanced weights
print("\n8c. Logistic Regression Classifier (balanced)...")
start_time = time.time()
lr_model = LogisticRegression(
    max_iter=1000,
    class_weight='balanced',  # Handle imbalance
    random_state=42,
    n_jobs=-1,
    verbose=0
)
lr_model.fit(X_train, y_train)
lr_time = time.time() - start_time
y_pred_lr = lr_model.predict(X_test)
accuracy_lr = accuracy_score(y_test, y_pred_lr)
balanced_acc_lr = balanced_accuracy_score(y_test, y_pred_lr)
print(f"   ✓ Training time: {lr_time:.2f}s")
print(f"   ✓ Accuracy: {accuracy_lr:.4f}")
print(f"   ✓ Balanced Accuracy: {balanced_acc_lr:.4f}")

# ============================================================================
# 9. MODEL COMPARISON
# ============================================================================
print("\n" + "=" * 80)
print("MODEL COMPARISON")
print("=" * 80)

models_comparison = {
    'Model': ['Random Forest', 'LightGBM', 'Logistic Regression'],
    'Train Time (s)': [rf_time, lgbm_time, lr_time],
    'Accuracy': [accuracy_rf, accuracy_lgbm, accuracy_lr],
    'Balanced Acc': [balanced_acc_rf, balanced_acc_lgbm, balanced_acc_lr],
    'Precision': [
        precision_score(y_test, y_pred_rf, average='weighted', zero_division=0),
        precision_score(y_test, y_pred_lgbm, average='weighted', zero_division=0),
        precision_score(y_test, y_pred_lr, average='weighted', zero_division=0)
    ],
    'Recall': [
        recall_score(y_test, y_pred_rf, average='weighted', zero_division=0),
        recall_score(y_test, y_pred_lgbm, average='weighted', zero_division=0),
        recall_score(y_test, y_pred_lr, average='weighted', zero_division=0)
    ],
    'F1-Score': [
        f1_score(y_test, y_pred_rf, average='weighted', zero_division=0),
        f1_score(y_test, y_pred_lgbm, average='weighted', zero_division=0),
        f1_score(y_test, y_pred_lr, average='weighted', zero_division=0)
    ]
}

comparison_df = pd.DataFrame(models_comparison)
comparison_df = comparison_df.round(4)

print("\n" + comparison_df.to_string(index=False))

best_model_idx = comparison_df['Balanced Acc'].idxmax()
print(f"\n✓ Best Model: {comparison_df.loc[best_model_idx, 'Model']} " +
      f"(Balanced Acc: {comparison_df.loc[best_model_idx, 'Balanced Acc']:.4f})")

# ============================================================================
# 10. PER-CLASS PERFORMANCE
# ============================================================================
print("\n" + "=" * 80)
print("PER-CLASS CLASSIFICATION REPORTS")
print("=" * 80)

models = [
    ('Random Forest', y_pred_rf),
    ('LightGBM', y_pred_lgbm),
    ('Logistic Regression', y_pred_lr)
]

for model_name, y_pred in models:
    print(f"\n{model_name}:")
    print(classification_report(y_test, y_pred,
                                target_names=[f'Cluster {i}' for i in range(10)],
                                zero_division=0))

# ============================================================================
# 11. SAVE RESULTS
# ============================================================================
print("\n" + "=" * 80)
print("SAVING RESULTS")
print("=" * 80)

# Save clustered data
output_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()
output_path = os.path.join(output_dir, "wallet_cluster_results_balanced.csv")
df_wallet_clean.to_csv(output_path, index=False)
print(f"\n✓ Clustered data saved to: {output_path}")

# Save cluster analysis
cluster_analysis_path = os.path.join(output_dir, "cluster_analysis_k10.csv")
cluster_df.to_csv(cluster_analysis_path, index=False)
print(f"✓ Cluster analysis saved to: {cluster_analysis_path}")

# Save model comparison
model_comparison_path = os.path.join(output_dir, "model_comparison_balanced.csv")
comparison_df.to_csv(model_comparison_path, index=False)
print(f"✓ Model comparison saved to: {model_comparison_path}")

# ============================================================================
# 12. CLEANUP
# ============================================================================
session.close()
print("\n" + "=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80)
print("\n✓ Improvements applied:")
print("  - k=10 clusters (better granularity)")
print("  - class_weight='balanced' (handles imbalance)")
print("  - Balanced accuracy metric (better for imbalanced data)")
print("  - Detailed per-cluster analysis")
print("\n✓ This approach better handles the imbalanced wallet distribution!")
