# WalletScope: Classification of Blockchain Wallets Using Machine Learning

**Author:** Venkata Sai Sushma Emmadi
**Semester:** Fall 2025
**Course:** DATA 606 - Capstone in Data Science
**University:** University of Maryland, Baltimore County (UMBC)
**Advisor:** Dr. Chaojie (Jay) Wang

---

## Quick Links

| Resource | Link |
|----------|------|
| GitHub Repository | [View Repository](https://github.com/sushma-0709/UMBC-DATA606-Capstone) |
| YouTube Presentation | [Watch the Video Presentation](#) |
| PowerPoint Presentation | [View Final PPT](https://docs.google.com/presentation/d/10uUQ2bS4B_PLOdnNf7CO5e2t-WPFfvhA/edit?usp=sharing&ouid=100176323177110534783&rtpof=true&sd=true) |
| LinkedIn | [Venkata Sai Sushma Emmadi](https://www.linkedin.com/in/sushma-evs-514687170/) |

---

## Table of Contents

1. [Introduction](#introduction)
2. [Problem Statement](#problem-statement)
3. [Dataset Overview](#dataset-overview)
4. [System Architecture](#system-architecture)
5. [Technology Stack](#technology-stack)
6. [Data Pipeline](#data-pipeline)
7. [Feature Engineering](#feature-engineering)
8. [Machine Learning Models](#machine-learning-models)
9. [Results & Evaluation](#results--evaluation)
10. [Installation & Setup](#installation--setup)
11. [Project Structure](#project-structure)
12. [Future Enhancements](#future-enhancements)
13. [References](#references)

---

## Introduction

Blockchain technology has revolutionized financial transactions by enabling decentralized, transparent, and secure transfers of digital assets. As cryptocurrency adoption grows, understanding the behavior patterns of blockchain wallets becomes increasingly important for market analysis, fraud detection, and regulatory compliance.

A **blockchain wallet** is a digital address used to send, receive, and store cryptocurrency. Wallets exhibit diverse behavioral patterns—some belong to individual users making occasional transactions, others to large holders (commonly called "Whales") who can influence market stability, and some to Exchanges or Merchants that process high volumes of transactions for many users.

This project implements a **data-driven wallet classification system** that leverages the modern data stack (Snowflake, dbt, Python) and machine learning algorithms to automatically categorize Litecoin wallets based on their on-chain activity patterns.

---

## Problem Statement

### Why Wallet Classification Matters

1. **Financial Insights**: Identifying Whales helps predict potential market movements and assess market stability
2. **Fraud Detection**: Unusual wallet activity patterns can indicate money laundering, scams, or other illicit activities
3. **Market Analysis**: Understanding the distribution of wallet types provides insights into cryptocurrency adoption and usage patterns
4. **Regulatory Compliance**: Exchanges and financial institutions need to identify and monitor high-value or high-risk wallets

### Challenge

The primary challenge is that blockchain data is **unlabeled**—there is no ground truth indicating which wallets belong to individuals, exchanges, or whales. This project addresses this by:
- Developing a rule-based labeling strategy using domain knowledge
- Engineering meaningful features from raw blockchain data
- Training supervised machine learning models for automated classification

---

## Dataset Overview

### Data Source

The project uses **Blockchair's Litecoin blockchain data dumps**, which provide comprehensive, daily-updated blockchain data. Blockchair is a trusted provider of blockchain analytics data.

### Dataset Statistics

| Data Type | Approximate Size | Description |
|-----------|------------------|-------------|
| **Addresses** | ~2 GB | Wallet addresses and current balances |
| **Blocks** | ~100 MB | Block metadata including timestamps, mining info |
| **Transactions** | ~10 GB | Transaction records with inputs/outputs |
| **Inputs** | ~8 GB | Source of funds for each transaction |
| **Outputs** | ~9 GB | Destination of funds for each transaction |

### Data Schema Details

#### 1. Addresses
| Column | Description |
|--------|-------------|
| `address` | Unique wallet identifier (string) |
| `balance` | Current balance in litoshis (1 LTC = 100,000,000 litoshis) |

#### 2. Blocks
| Column | Description |
|--------|-------------|
| `id` | Block height/number |
| `hash` | Unique block identifier |
| `time` | Block creation timestamp |
| `transaction_count` | Number of transactions in block |
| `difficulty` | Mining difficulty |
| `fee_total` | Total fees collected in block |
| `reward` | Mining reward |
| `guessed_miner` | Probable mining pool |

#### 3. Transactions
| Column | Description |
|--------|-------------|
| `hash` | Unique transaction identifier |
| `block_id` | Parent block |
| `time` | Transaction timestamp |
| `input_count` / `output_count` | Number of inputs/outputs |
| `input_total` / `output_total` | Total value of inputs/outputs |
| `fee` | Transaction fee |

#### 4. Inputs
| Column | Description |
|--------|-------------|
| `transaction_hash` | Parent transaction |
| `recipient` | Source wallet address |
| `value` | Amount sent |

#### 5. Outputs
| Column | Description |
|--------|-------------|
| `transaction_hash` | Parent transaction |
| `recipient` | Destination wallet address |
| `value` | Amount received |

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                     WalletScope Architecture                                 │
└─────────────────────────────────────────────────────────────────────────────┘

    ┌──────────────┐
    │  Blockchair  │     Data Source: Daily blockchain dumps
    │    (Extract) │     - Addresses, Blocks, Transactions
    └──────┬───────┘     - Inputs, Outputs
           │
           ▼
    ┌──────────────┐
    │   Snowflake  │     Cloud Data Warehouse
    │   (Storage)  │     - Raw schema: Landing zone
    └──────┬───────┘     - Scalable cloud storage
           │
           ▼
    ┌──────────────┐     SQL Transformation Engine
    │     dbt      │     ┌─────────────────────────────────────────┐
    │ (Transform)  │────▶│  Staging → Intermediate → Core → Analytics │
    └──────┬───────┘     └─────────────────────────────────────────┘
           │
           ▼
    ┌──────────────┐
    │   Python     │     Machine Learning Pipeline
    │     (ML)     │     - Feature scaling (StandardScaler)
    │              │     - KMeans clustering (k=10)
    └──────┬───────┘     - Classification models
           │
           ▼
    ┌──────────────┐
    │   Results    │     Output Artifacts
    │   (Export)   │     - Wallet classifications
    └──────────────┘     - Model metrics
```

### Modern Data Stack Components

| Layer | Technology | Purpose |
|-------|------------|---------|
| **Extract** | Blockchair API | Download raw blockchain data dumps |
| **Storage** | Snowflake | Cloud data warehouse for scalable storage |
| **Transform** | dbt | SQL-based transformation and feature engineering |
| **ML** | Python (scikit-learn, LightGBM) | Model training and classification |
| **Visualization** | Power BI | Dashboard and reporting |
| **Orchestration** | Airflow | Pipeline automation and scheduling |

---

## Technology Stack

| Category | Technology | Version | Purpose |
|----------|------------|---------|---------|
| **Language** | Python | 3.10+ | Core development |
| **Data Warehouse** | Snowflake | Latest | Cloud storage and compute |
| **Transformation** | dbt-core | 1.10.13 | SQL modeling |
| **ML Framework** | scikit-learn | 1.7.2 | Machine learning |
| **Gradient Boosting** | LightGBM | Latest | High-performance classification |
| **Data Processing** | pandas | 2.3.3 | Data manipulation |
| **Numerical Computing** | NumPy | 2.3.4 | Array operations |
| **Snowflake Connector** | snowflake-snowpark-python | 1.42.0 | Snowflake integration |
| **Configuration** | python-dotenv | 1.2.1 | Environment management |

---

## Data Pipeline

### dbt Transformation Layers

The project implements a layered transformation approach following data engineering best practices:

```
RAW DATA                    STAGING              INTERMEDIATE           CORE                 ANALYTICS
─────────                   ───────              ────────────           ────                 ─────────
addresses.tsv      ──▶    stg_addresses    ┐
                                            │
blocks.tsv         ──▶    stg_blocks       │
                                            ├──▶  int_wallet      ──▶  wallet_activity   ┐
transactions.tsv   ──▶    stg_transactions │      _transfers                              │
                                            │                                              ├──▶  WALLET_
inputs.tsv         ──▶    stg_inputs       │                                              │      CLASSIFICATION
                                            │                          transaction_       │
outputs.tsv        ──▶    stg_outputs      ┘                          summary            │
                                                                                          │
                                                                       fct_wallet_       ─┘
                                                                       daily_features
```

### Layer Descriptions

| Layer | Materialization | Purpose |
|-------|-----------------|---------|
| **Staging** | Views | Clean and standardize raw data, rename columns |
| **Intermediate** | Views | Combine related tables, add business logic |
| **Core** | Tables | Business-level aggregations (wallet activity, summaries) |
| **Analytics** | Tables | Final feature tables for ML consumption |

### Key dbt Models

#### `int_wallet_transfers.sql`
Combines incoming and outgoing transactions with direction flag:
- Incoming: Funds received (from outputs table)
- Outgoing: Funds sent (from inputs table)

#### `wallet_activity.sql`
Aggregates wallet-level metrics:
- Total sent and received amounts
- Net balance calculations
- Transaction counts

#### `fct_wallet_daily_features.sql`
Daily wallet activity features:
- Daily transaction counts
- Daily sent/received amounts
- Active flag (1 if transactions, 0 otherwise)

#### `wallet_classification.sql`
Final feature table for ML:
- Lifecycle metrics (active days, first/last seen)
- Engagement metrics (avg transactions per day)
- Volume metrics (total sent/received)

---

## Feature Engineering

### Feature Set (8 Primary Features)

| Feature | Source | Type | Description |
|---------|--------|------|-------------|
| `ACTIVE_DAYS` | Computed | Derived | Days between first and last activity |
| `TOTAL_TX_COUNT` | transactions | Derived | Total transactions (incoming + outgoing) |
| `TOTAL_OUTGOING_TX` | inputs | Derived | Number of outgoing transactions |
| `LIFETIME_SENT_LTC` | inputs | Derived | Total LTC sent over wallet lifetime |
| `LIFETIME_RECEIVED_LTC` | outputs | Derived | Total LTC received over wallet lifetime |
| `CURRENT_BALANCE_LTC` | addresses | Direct | Current wallet balance |
| `AVG_SENT_PER_DAY` | Computed | Derived | Average daily sent amount |
| `AVG_RECEIVED_PER_DAY` | Computed | Derived | Average daily received amount |

### Extended Feature Set (From Proposal)

| Feature | Description |
|---------|-------------|
| `wallet_address` | Primary key identifier |
| `balance` | Current wallet balance |
| `first_seen` / `last_seen` | Activity timeframe |
| `num_incoming_tx` / `num_outgoing_tx` | Transaction counts by direction |
| `total_received` / `total_sent` | Volume by direction |
| `avg_incoming_value` / `avg_outgoing_value` | Average transaction values |
| `inputs_per_tx_avg` / `outputs_per_tx_avg` | Transaction complexity |
| `unique_counterparties` | Number of distinct wallet interactions |
| `fee_ratio` | Fees as percentage of transaction volume |

---

## Machine Learning Models

### Target Variable: Wallet Categories

The project classifies wallets into three categories:

| Category | Description | Identification Criteria |
|----------|-------------|------------------------|
| **Whale** | High-value holders | Top 1% by balance AND minimum 10 transactions |
| **Exchange/Merchant** | High-volume processors | Top 5% by transaction count OR top 5% by received amount |
| **Individual** | Regular users | All other wallets |

### Labeling Strategy

Since blockchain data lacks ground truth labels, a rule-based approach assigns initial labels:

```python
# Classification thresholds
WHALE_BALANCE_PERCENTILE = 0.99        # Top 1% by balance
WHALE_MIN_TX = 10                       # Must have activity (not dormant)
EXCHANGE_TX_PERCENTILE = 0.95          # Top 5% by transaction volume
EXCHANGE_RECEIVED_PERCENTILE = 0.95    # OR top 5% by received amount
EXCHANGE_MIN_BALANCE = 0.1             # Must have some balance
```

### Model Training Pipeline

```
┌────────────────┐     ┌────────────────┐     ┌────────────────┐
│  Load Data     │────▶│  Clean Data    │────▶│  Scale Features│
│  (Snowflake)   │     │  (dropna)      │     │  (StandardScaler)
└────────────────┘     └────────────────┘     └───────┬────────┘
                                                       │
                                                       ▼
┌────────────────┐     ┌────────────────┐     ┌────────────────┐
│  Evaluate      │◀────│  Train Models  │◀────│  Train/Test    │
│  & Compare     │     │  (3 algorithms)│     │  Split (80/20) │
└────────────────┘     └────────────────┘     └────────────────┘
```

### Algorithms Implemented

#### 1. Random Forest Classifier
```python
RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    class_weight='balanced',
    random_state=42
)
```

#### 2. LightGBM Classifier
```python
LGBMClassifier(
    n_estimators=100,
    num_leaves=31,
    learning_rate=0.1,
    class_weight='balanced',
    random_state=42
)
```

#### 3. Logistic Regression
```python
LogisticRegression(
    max_iter=1000,
    class_weight='balanced',
    random_state=42
)
```

### Handling Class Imbalance

The wallet distribution is inherently imbalanced (most wallets are individuals). Strategies employed:

1. **Balanced Class Weights**: All models use `class_weight='balanced'`
2. **Balanced Accuracy Metric**: Prioritized over standard accuracy
3. **Stratified Split**: Train/test split maintains class proportions
4. **KMeans Clustering**: Used for additional pattern discovery (k=10)

---

## Results & Evaluation

### Evaluation Metrics

| Metric | Description |
|--------|-------------|
| **Accuracy** | Overall correct predictions |
| **Balanced Accuracy** | Average recall across classes (handles imbalance) |
| **Precision** | True positives / Predicted positives |
| **Recall** | True positives / Actual positives |
| **F1-Score** | Harmonic mean of precision and recall |
| **Calinski-Harabasz Score** | Cluster separation quality |

### Model Comparison Framework

```python
models_comparison = {
    'Model': ['Random Forest', 'LightGBM', 'Logistic Regression'],
    'Accuracy': [...],
    'Balanced Acc': [...],
    'Precision': [...],
    'Recall': [...],
    'F1-Score': [...]
}
```

### Key Outputs

- `wallet_cluster_results_balanced.csv`: Complete wallet classifications
- `cluster_analysis_k10.csv`: Cluster characteristics summary
- `model_comparison_balanced.csv`: Model performance metrics

---

## Installation & Setup

### Prerequisites

- Python 3.8 or higher
- Snowflake account with appropriate credentials
- dbt installed and configured
- Access to Blockchair data or downloaded dumps

### Setup Instructions

```bash
# 1. Clone the repository
git clone https://github.com/sushma-0709/UMBC-DATA606-Capstone.git
cd UMBC-DATA606-Capstone

# 2. Create virtual environment
python -m venv venv

# 3. Activate virtual environment
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# 4. Install dependencies
pip install -r requirments.txt

# 5. Configure environment variables
# Create .env file with Snowflake credentials:
# SNOWFLAKE_ACCOUNT=your_account
# SNOWFLAKE_USER=your_user
# SNOWFLAKE_PASSWORD=your_password
# SNOWFLAKE_WAREHOUSE=your_warehouse
# SNOWFLAKE_DATABASE=your_database
# SNOWFLAKE_SCHEMA=your_schema
# SNOWFLAKE_TOML=/path/to/snowflakecli.toml

# 6. Download blockchain data (optional)
python Scripts/download_data.py

# 7. Run dbt transformations
cd blockchair_wallet_classification
dbt run

# 8. Run ML classification
python notebooks/ml_classification.py
```

---

## Project Structure

```
UMBC-DATA606-Capstone/
├── README.md                              # Project overview
├── LICENSE                                # Apache 2.0 License
├── requirments.txt                        # Python dependencies (125 packages)
├── config_loader.py                       # Snowflake configuration utility
├── snowflakecli.toml                      # Snowflake connection profiles
├── .gitignore                             # Git ignore rules
│
├── Scripts/                               # Data loading scripts
│   ├── download_data.py                   # Blockchair data downloader
│   └── load_data.py                       # Snowflake connection test
│
├── sql/                                   # Raw SQL setup scripts
│   ├── addresses.sql                      # Create addresses table
│   ├── blocks.sql                         # Create blocks table
│   ├── transactions.sql                   # Create transactions table
│   ├── inputs.sql                         # Create inputs table
│   ├── outputs.sql                        # Create outputs table
│   ├── file_format.sql                    # Snowflake file format
│   ├── stage.sql                          # Create Snowflake stage
│   ├── put_data.sql                       # Upload data to stage
│   └── copy_to_table.sql                  # Load data into tables
│
├── blockchair_wallet_classification/      # dbt project
│   ├── dbt_project.yml                    # dbt configuration
│   ├── models/
│   │   ├── staging/                       # Layer 1: Raw → Clean
│   │   │   ├── stg_addresses.sql
│   │   │   ├── stg_blocks.sql
│   │   │   ├── stg_inputs.sql
│   │   │   ├── stg_outputs.sql
│   │   │   ├── stg_transactions.sql
│   │   │   └── schema.yml
│   │   ├── intermediate/                  # Layer 2: Combine & Join
│   │   │   ├── int_wallet_transfers.sql
│   │   │   └── schema.yml
│   │   ├── core/                          # Layer 3: Business Logic
│   │   │   ├── wallet_activity.sql
│   │   │   ├── transaction_summary.sql
│   │   │   └── schema.yml
│   │   └── analytics/                     # Layer 4: ML Features
│   │       ├── fct_wallet_daily_features.sql
│   │       ├── wallet_classification.sql
│   │       └── schema.yml
│   └── macros/
│       └── generate_schema_name.sql
│
├── notebooks/                             # Analysis & ML
│   ├── ml_classification.py               # Main ML pipeline (389 lines)
│   ├── ml_wallet_classification.ipynb     # Comprehensive notebook
│   ├── addresses.ipynb                    # EDA: Addresses
│   ├── blocks.ipynb                       # EDA: Blocks
│   ├── transactions.ipynb                 # EDA: Transactions
│   ├── inputs.ipynb                       # EDA: Inputs
│   └── outputs.ipynb                      # EDA: Outputs
│
├── data/                                  # Data directory (gitignored)
│   └── README.md
│
├── docs/                                  # Documentation
│   ├── Proposal.md                        # Project proposal
│   ├── report.md                          # Final report (this file)
│   └── Resume.md                          # Author resume
│
└── utils/
    └── config_loader.py                   # Config utilities
```

---

## Future Enhancements

### Short-Term Improvements

- [ ] **GPU Acceleration**: Leverage RAPIDS for faster processing of large datasets
- [ ] **Data Export**: Add CSV/Excel export functionality for analysis results
- [ ] **Dashboard Integration**: Connect Power BI for real-time visualization
- [ ] **Automated Retraining**: Schedule periodic model updates with Airflow

### Long-Term Roadmap

- [ ] **Multi-Chain Support**: Extend to Bitcoin, Ethereum, and other blockchains
- [ ] **Real-Time Classification**: Implement streaming data pipeline for live wallet monitoring
- [ ] **API Service**: Deploy as REST API for integration with other systems
- [ ] **Anomaly Detection**: Add unsupervised learning for suspicious activity detection
- [ ] **Network Analysis**: Incorporate graph-based features (wallet clusters, communities)
- [ ] **Cross-Chain Analysis**: Track wallet behavior across multiple blockchains

---

## References

1. **Blockchair Documentation**: [https://blockchair.com/api/docs](https://blockchair.com/api/docs)
2. **Snowflake Documentation**: [https://docs.snowflake.com/](https://docs.snowflake.com/)
3. **dbt Documentation**: [https://docs.getdbt.com/](https://docs.getdbt.com/)
4. **scikit-learn Documentation**: [https://scikit-learn.org/stable/](https://scikit-learn.org/stable/)
5. **LightGBM Documentation**: [https://lightgbm.readthedocs.io/](https://lightgbm.readthedocs.io/)

---

## License

This project is licensed under the **Apache License 2.0** - see the [LICENSE](../LICENSE) file for details.

---

## Acknowledgments

- **Dr. Chaojie (Jay) Wang** - Project Advisor
- **UMBC Data Science Department** - Academic support and resources
- **Blockchair** - Providing comprehensive blockchain data

---

*This project was developed as part of the DATA 606 Capstone in Data Science course at the University of Maryland, Baltimore County (UMBC).*

