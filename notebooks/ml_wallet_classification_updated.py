"""
Wallet Classification ML Pipeline - Updated with Fixed Data
Uses the corrected ANALYTICS schema with zero values instead of NULLs
"""

import pandas as pd
import numpy as np
from snowflake.snowpark import Session
import tomllib
import os
from dotenv import load_dotenv
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Load environment
load_dotenv()

print("=" * 80)
print("WALLET CLASSIFICATION ML PIPELINE - UPDATED")
print("=" * 80)

# Connect to Snowflake
print("\n[1/7] Connecting to Snowflake...")
with open(os.environ["SNOWFLAKE_TOML"], "rb") as f:
    config = tomllib.load(f)

wallet_classification_config = config["connections"]["wallet_classification"]
conn_params = {k: os.path.expandvars(v) for k, v in wallet_classification_config.items()}
session = Session.builder.configs(conn_params).create()

print(f"✓ Connected - Role: {session.sql('SELECT CURRENT_ROLE()').collect()[0][0]}")
print(f"✓ Current Schema: {session.sql('SELECT CURRENT_SCHEMA()').collect()[0][0]}")

# Use the correct ANALYTICS schema (not RAW_analytics)
print("\n[2/7] Switching to ANALYTICS schema...")
session.use_schema("ANALYTICS")
print(f"✓ Now using schema: {session.sql('SELECT CURRENT_SCHEMA()').collect()[0][0]}")

# Load data
print("\n[3/7] Loading wallet classification data...")
df = session.table("WALLET_CLASSIFICATION")
df_wallet = df.to_pandas()
print(f"✓ Loaded {len(df_wallet):,} wallets with {len(df_wallet.columns)} features")

# Verify the fix - check for NULL values
print("\n[4/7] Verifying data quality (NULL check)...")
null_counts = df_wallet.isnull().sum()
total_nulls = null_counts.sum()
print(f"Total NULL values across all columns: {total_nulls}")

if null_counts['CURRENT_BALANCE_LTC'] > 0:
    print(f"⚠ WARNING: Found {null_counts['CURRENT_BALANCE_LTC']} NULL values in CURRENT_BALANCE_LTC")
    print("Applying fallback: filling NULLs with 0")
    df_wallet['CURRENT_BALANCE_LTC'] = df_wallet['CURRENT_BALANCE_LTC'].fillna(0)
else:
    print("✓ No NULL values in CURRENT_BALANCE_LTC - fix successful!")

# Show distribution of zero balances
zero_balance_count = (df_wallet['CURRENT_BALANCE_LTC'] == 0).sum()
positive_balance_count = (df_wallet['CURRENT_BALANCE_LTC'] > 0).sum()
print(f"\nBalance Distribution:")
print(f"  Zero balance wallets: {zero_balance_count:,} ({zero_balance_count/len(df_wallet)*100:.1f}%)")
print(f"  Positive balance wallets: {positive_balance_count:,} ({positive_balance_count/len(df_wallet)*100:.1f}%)")

# Display basic statistics
print("\n" + "=" * 80)
print("DATASET STATISTICS")
print("=" * 80)
print(df_wallet.describe())

# Define features for ML
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

print(f"\n[5/7] Preparing features for ML...")
print(f"Selected features: {features}")

# Prepare feature matrix
X = df_wallet[features].copy()

# Check for any remaining NaN values and handle them
remaining_nans = X.isnull().sum().sum()
if remaining_nans > 0:
    print(f"⚠ Found {remaining_nans} NaN values in features, filling with median")
    X = X.fillna(X.median())
else:
    print("✓ No NaN values in feature matrix")

print(f"✓ Feature matrix shape: {X.shape}")

# Scale features
print("\n[6/7] Training models...")
print("\n--- K-Means Clustering ---")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# KMeans clustering
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
cluster_labels = kmeans.fit_predict(X_scaled)
df_wallet['cluster'] = cluster_labels

# Analyze clusters
print("\nCluster Statistics:")
cluster_stats = df_wallet.groupby('cluster')[features].mean()
print(cluster_stats.round(2))

# Map clusters to meaningful names based on activity
cluster_map = {
    0: 'High Activity',
    1: 'Low Activity',
    2: 'Dormant'
}

df_wallet['wallet_category'] = df_wallet['cluster'].map(cluster_map)

print("\nCluster Distribution:")
for category, count in df_wallet['wallet_category'].value_counts().items():
    percentage = (count / len(df_wallet)) * 100
    print(f"  {category}: {count:,} wallets ({percentage:.1f}%)")

# Random Forest Classification
print("\n--- Random Forest Classifier ---")
X_classification = df_wallet[features].copy()
y_classification = df_wallet['cluster'].copy()

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X_classification, y_classification,
    test_size=0.2,
    random_state=42,
    stratify=y_classification
)

print(f"Training set size: {len(X_train):,}")
print(f"Test set size: {len(X_test):,}")

# Scale features
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train Random Forest
rf_classifier = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    max_depth=10,
    min_samples_split=5,
    min_samples_leaf=2,
    n_jobs=-1
)

print("\nTraining Random Forest...")
rf_classifier.fit(X_train_scaled, y_train)

# Predictions
y_pred = rf_classifier.predict(X_test_scaled)

# Evaluation
print("\n[7/7] Evaluating model performance...")
print("\n" + "=" * 80)
print("MODEL PERFORMANCE RESULTS")
print("=" * 80)

train_accuracy = rf_classifier.score(X_train_scaled, y_train)
test_accuracy = rf_classifier.score(X_test_scaled, y_test)

print(f"\nTraining Accuracy: {train_accuracy:.4f}")
print(f"Testing Accuracy: {test_accuracy:.4f}")

print("\n" + "-" * 80)
print("CLASSIFICATION REPORT")
print("-" * 80)
unique_classes = sorted(list(set(y_test.unique()) | set(y_pred)))
target_names = [cluster_map[i] for i in unique_classes]
print(classification_report(y_test, y_pred, labels=unique_classes, target_names=target_names))

# Feature Importance
print("\n" + "-" * 80)
print("FEATURE IMPORTANCE")
print("-" * 80)
feature_importance = pd.DataFrame({
    'feature': features,
    'importance': rf_classifier.feature_importances_
}).sort_values('importance', ascending=False)
print(feature_importance.to_string(index=False))

# Confusion Matrix
print("\n" + "-" * 80)
print("CONFUSION MATRIX")
print("-" * 80)
cm = confusion_matrix(y_test, y_pred)
cm_df = pd.DataFrame(cm,
                      index=['High Activity', 'Low Activity', 'Dormant'],
                      columns=['High Activity', 'Low Activity', 'Dormant'])
print(cm_df)

# Save results
print("\n" + "=" * 80)
print("SAVING RESULTS")
print("=" * 80)

# Save classified data
output_file = '/root/UMBC-DATA606-Capstone/notebooks/wallet_classification_results.csv'
df_wallet[['ADDRESS', 'cluster', 'wallet_category'] + features].to_csv(output_file, index=False)
print(f"✓ Results saved to: {output_file}")

# Save model metrics
metrics_file = '/root/UMBC-DATA606-Capstone/notebooks/model_metrics.txt'
with open(metrics_file, 'w') as f:
    f.write("WALLET CLASSIFICATION MODEL METRICS\n")
    f.write("=" * 80 + "\n\n")
    f.write(f"Total Wallets Analyzed: {len(df_wallet):,}\n")
    f.write(f"Training Set Size: {len(X_train):,}\n")
    f.write(f"Test Set Size: {len(X_test):,}\n\n")
    f.write(f"Training Accuracy: {train_accuracy:.4f}\n")
    f.write(f"Testing Accuracy: {test_accuracy:.4f}\n\n")
    f.write("Cluster Distribution:\n")
    for category, count in df_wallet['wallet_category'].value_counts().items():
        percentage = (count / len(df_wallet)) * 100
        f.write(f"  {category}: {count:,} ({percentage:.1f}%)\n")
    f.write("\n")
    f.write(feature_importance.to_string(index=False))
print(f"✓ Metrics saved to: {metrics_file}")

print("\n" + "=" * 80)
print("PIPELINE COMPLETED SUCCESSFULLY!")
print("=" * 80)

# Close session
session.close()
print("\n✓ Snowflake session closed")
