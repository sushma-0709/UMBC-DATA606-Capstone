# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**WalletScope** is a UMBC Data Science Master's Capstone project that classifies blockchain wallets using Litecoin blockchain data. The project identifies wallet categories (Small/Individual, Whale, and Exchange/Merchant wallets) to provide financial insights, support fraud detection, and enable market analysis.

## Architecture & Technology Stack

### High-Level Architecture

The project implements a modern **ELT/ETL pipeline** with the following flow:

```
Blockchair (Data Source)
    ↓
Scripts/download_data.py (Extract)
    ↓
Snowflake Cloud Data Warehouse (Raw Schema)
    ↓
dbt Transformations + SQL (Transform & Feature Engineering)
    ↓
ML Classification Models (XGBoost, Random Forest)
    ↓
Power BI Dashboards (Visualization)
```

### Key Technologies

- **Data Warehouse:** Snowflake (with dbt for transformations)
- **Programming:** Python 3.x with Jupyter notebooks
- **ML Frameworks:** XGBoost, Random Forest (scikit-learn)
- **Orchestration:** Apache Airflow (planned)
- **CLI Tools:** Snowflake CLI, dbt CLI
- **Cloud Storage:** AWS (boto3)

### Database Schema

Raw tables in `LITECOIN.RAW` schema:
- **blocks** - Block metadata, timestamps, hashing info (~100 MB)
- **transactions** - Transaction details, fees, input/output counts (~10 GB)
- **inputs** - Source addresses and values (~8 GB)
- **outputs** - Destination addresses and values (~9 GB)
- **addresses** - Litecoin addresses and balances (~2 GB)

## Project Structure

```
Scripts/                 # ETL and data pipeline scripts
├── download_data.py    # Download from Blockchair API
└── load_data.py        # Load data to Snowflake

sql/                    # SQL scripts for schema setup
├── file_format.sql    # Define TSV file format
├── stage.sql          # Create Snowflake stage
├── copy_to_table.sql  # Load data into tables
└── {table_name}.sql   # CREATE TABLE statements

notebooks/             # Jupyter notebooks for EDA
├── addresses.ipynb
├── blocks.ipynb
├── transactions.ipynb
├── inputs.ipynb
└── outputs.ipynb

data/litecoin/        # Blockchain data storage
├── addresses/
├── blocks/
├── transactions/
├── inputs/
└── outputs/

utils/                 # Utility modules
└── config_loader.py  # Snowflake connection config

.dbt/                 # dbt configuration
├── profiles.yml     # Connection profiles
└── .user.yml        # User settings

docs/                 # Project documentation
├── Proposal.md      # Comprehensive project proposal
└── Resume.md        # Student resume
```

## Common Development Commands

### Environment Setup

```bash
# Activate virtual environment
source .venv/bin/activate

# Install/update dependencies
pip install -r requirments.txt

# Create/verify dbt profiles
dbt debug --profiles-dir .dbt
```

### Data Pipeline

```bash
# Download latest data from Blockchair
python Scripts/download_data.py

# Load downloaded data to Snowflake
python Scripts/load_data.py

# Run SQL setup scripts in Snowflake
# Execute in order: file_format.sql → stage.sql → {table_name}.sql → put_data.sql → copy_to_table.sql
```

### dbt Commands (for transformations)

```bash
# Parse the dbt project
dbt parse --profiles-dir .dbt

# Run dbt models
dbt run --profiles-dir .dbt

# Run tests
dbt test --profiles-dir .dbt

# Generate documentation
dbt docs generate --profiles-dir .dbt

# Debug dbt connection
dbt debug --profiles-dir .dbt
```

### Jupyter Notebooks

```bash
# Start Jupyter Lab
jupyter lab

# Then open notebooks in notebooks/ directory for EDA
```

### Snowflake CLI

```bash
# Test connection
snow connection test --connection-name litecoin

# Execute SQL scripts
snow sql -f sql/file_format.sql --connection-name litecoin
```

## Configuration Files

### Environment Variables (.env)

Required Snowflake credentials:
- `SNOWFLAKE_ACCOUNT` - Account identifier (e.g., YAFSHSP-YOB78287)
- `SNOWFLAKE_USER` - Username for authentication
- `SNOWFLAKE_PASSWORD` - Password (stored securely, not in git)
- `SNOWFLAKE_ROLE` - Role (e.g., ACCOUNTADMIN)
- `SNOWFLAKE_WAREHOUSE` - Warehouse name (e.g., BLOCKCHAIR)
- `SNOWFLAKE_DATABASE` - Database (e.g., LITECOIN)
- `SNOWFLAKE_SCHEMA` - Schema (e.g., RAW)

### Snowflake CLI (snowflakecli.toml)

- Two connection profiles: `default` and `litecoin`
- Logging configured to `logs/snowflake-cli.log`
- Used for direct Snowflake CLI operations

### dbt Profiles (.dbt/profiles.yml)

- Configured for Snowflake adapter
- Uses environment variables for sensitive credentials
- Supports multiple dbt profiles/targets if needed

## Machine Learning Architecture

### Target Variable: wallet_category

- **Small/Individual** - Low balance, low transaction activity
- **Whale** - Very high balance (top 5% by balance)
- **Exchange/Merchant** - Very high transaction count (top 5% by activity)

### Feature Set (18 features)

- **Direct:** wallet_address, balance
- **Temporal:** first_seen, last_seen, activity_days
- **Incoming:** num_incoming_tx, total_received, avg_incoming_value
- **Outgoing:** num_outgoing_tx, total_sent, avg_outgoing_value
- **Aggregate:** total_tx, avg_tx_value, inputs_per_tx_avg, outputs_per_tx_avg
- **Network:** unique_counterparties
- **Cost:** fee_ratio

### Models

- **XGBoost** - Primary classifier (excellent on tabular data)
- **Random Forest** - Baseline model

## Key Dependencies

- **snowflake-connector-python** - Direct Snowflake connection
- **dbt-core** & **dbt-snowflake** - SQL transformation framework
- **numpy**, **pandas** - Data manipulation
- **requests** - API calls to Blockchair
- **PyYAML** - Configuration parsing
- **python-dotenv** - Environment variable management

## Important Notes

### Data Pipeline Flow

1. **Blockchair API** → Raw TSV files in `data/litecoin/{type}/`
2. **Snowflake Stage** → PUT compressed files to internal stage
3. **Raw Tables** → COPY data from stage (Snowflake)
4. **dbt Models** → Transform and create feature tables (future)
5. **ML Models** → Train on engineered features (planned)

### Snowflake Considerations

- Database: `LITECOIN`
- Raw Schema: `RAW` (staging area)
- Warehouse: `BLOCKCHAIR` (compute resource)
- Account: See `.env` file (not in git)
- Data format: TSV (Tab-Separated Values)

### dbt Workflow

- dbt is configured but transformation models need to be created
- Use `.dbt/profiles.yml` for connection configuration
- Place transformation models in `models/` directory (when created)
- Use Jinja2 templating for dynamic SQL generation

### Exploratory Analysis

Each notebook (addresses, blocks, transactions, inputs, outputs) explores:
- Data structure and shape
- Missing values and data quality
- Statistical distributions
- Key relationships and correlations
- Patterns relevant to wallet classification

## Repository Status

- ✅ Data extraction and Snowflake setup complete
- ✅ Raw tables created and data loaded
- ⏳ dbt transformation models (in progress)
- ⏳ Feature engineering and ML pipeline (planned)
- ⏳ Apache Airflow orchestration (planned)
- ⏳ Power BI dashboards (planned)

## Useful Documentation

- **Project Proposal:** See `docs/Proposal.md` for detailed problem statement, methodology, and expected outcomes
- **Data Source:** Blockchair API documentation for blockchain data schema
- **dbt Documentation:** https://docs.getdbt.com/
- **Snowflake Docs:** https://docs.snowflake.com/
