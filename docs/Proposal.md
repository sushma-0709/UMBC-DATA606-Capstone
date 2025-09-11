# WalletScope : Classification of Blockchain Wallets
####  Prepared for UMBC Data Science Master Degree Capstone by Dr Chaojie (Jay) Wang
#### Author Name : Venkata Sai Sushma Emmadi
#### Link to GitHub repo: https://github.com/sushma-0709/UMBC-DATA606-Capstone
#### LinkedIn profile: https://www.linkedin.com/in/sushma-evs-514687170/

## Project Overview
This project focuses on classifying blockchain wallets using data from Blockchair’s Litecoin dataset. A blockchain wallet is a digital address used to send, receive, and store cryptocurrency. Wallets vary in behavior—some belong to individual users, others to large holders (Whales), and some to Exchanges or Merchants that handle transactions for many users.

Classifying wallets matters because it provides financial insights by identifying Whales who can influence market stability, supports fraud detection by spotting unusual activity, and aids market analysis. Ultimately, wallet classification enables better understanding of transaction flows in blockchain ecosystems.

---

## Dataset Overview
Extracted from **Blockchair Litecoin dumps**, the dataset includes addresses, blocks, transactions, inputs, and outputs. Blockchair updates these data dumps on a **daily basis**, ensuring access to the latest blockchain activity for analysis.

Approximate file sizes:
- **Addresses:** ~2 GB
- **Blocks:** ~100 MB
- **Transactions:** ~10 GB
- **Inputs:** ~8 GB
- **Outputs:** ~9 GB

These sizes may vary as the blockchain grows and new data is added.

---

### 1. **Addresses**
Litecoin addresses are unique identifiers for wallets on the Litecoin blockchain. Each address can hold a balance, receive, and send transactions. Analyzing Litecoin address data enables classification of wallet types (such as personal, exchange, or miner) and tracking of activity patterns.

| Column     | Description                                                        |
|------------|--------------------------------------------------------------------|
| `address`  | The unique string representing a Litecoin wallet address.           |
| `balance`  | The current balance of the address, measured in litoshis (1 LTC = 100,000,000 litoshis). |

---

### 2. **Blocks**
Blocks are containers for groups of transactions on the blockchain. Each block includes metadata such as timestamps, hashes, and mining information, enabling verification and linking of transactions in a secure, chronological chain.

| Column | Description |
|--------|-------------|
| `id` | Block ID. |
| `hash` | Block hash. |
| `time` | Block mined timestamp. |
| `median_time` | Median of recent block times. |
| `size` | Block size (bytes). |
| `stripped_size` | Size excluding witness data. |
| `weight` | Block weight (SegWit). |
| `version`, `version_hex`, `version_bits` | Version metadata. |
| `merkle_root` | Merkle root hash. |
| `nonce` | Mining nonce. |
| `bits` | Difficulty target. |
| `difficulty` | Mining difficulty. |
| `chainwork` | Total chain work. |
| `coinbase_data_hex` | Miner-added data. |
| `transaction_count` | Transactions in block. |
| `witness_count` | Witness transactions. |
| `input_count`, `output_count` | Inputs/outputs count. |
| `input_total`, `input_total_usd` | Input totals (BTC, USD). |
| `output_total`, `output_total_usd` | Output totals. |
| `fee_total`, `fee_total_usd` | Total fees. |
| `fee_per_kb`, `fee_per_kb_usd` | Fee per KB. |
| `fee_per_kwu`, `fee_per_kwu_usd` | Fee per weight unit. |
| `cdd_total` | Coin Days Destroyed. |
| `generation`, `generation_usd` | Mining reward. |
| `reward`, `reward_usd` | Total block reward. |
| `guessed_miner` | Likely mining pool. |

---

### 3. **Transactions**
Transactions represent the transfer of cryptocurrency between addresses. Each transaction records inputs (sources of funds), outputs (destinations), fees, and other metadata, forming the core activity on the blockchain.

| Column | Description |
|--------|-------------|
| `block_id` | Block containing transaction. |
| `hash` | Transaction ID. |
| `time` | Transaction timestamp. |
| `size` | Transaction size (bytes). |
| `weight` | Transaction weight. |
| `version` | Format version. |
| `lock_time` | Minimum valid time. |
| `is_coinbase` | Coinbase transaction flag. |
| `has_witness` | SegWit usage. |
| `input_count`, `output_count` | Inputs/outputs count. |
| `input_total`, `input_total_usd` | Input totals. |
| `output_total`, `output_total_usd` | Output totals. |
| `fee`, `fee_usd` | Transaction fee. |
| `fee_per_kb`, `fee_per_kb_usd` | Fee per KB. |
| `fee_per_kwu`, `fee_per_kwu_usd` | Fee per weight unit. |
| `cdd_total` | Coin Days Destroyed. |

---

### 4. **Outputs**
Outputs specify the destination and amount of cryptocurrency sent in a transaction. Each output details the recipient address, value, and script, and can later be referenced as an input in a new transaction.

| Column | Description |
|--------|-------------|
| `block_id` | Block of output creation. |
| `transaction_hash` | Output's transaction. |
| `index` | Output index. |
| `time` | Output timestamp. |
| `value`, `value_usd` | Output value. |
| `recipient` | Recipient address. |
| `type` | Script type. |
| `script_hex` | Script in hex. |
| `is_from_coinbase` | From coinbase transaction. |
| `is_spendable` | Spendable flag. |
| `spending_block_id` | Block where spent. |
| `spending_transaction_hash` | Spending transaction. |
| `spending_index` | Spending index. |
| `spending_time` | Spending timestamp. |
| `spending_value_usd` | Value when spent. |
| `spending_sequence` | Spending sequence. |
| `spending_signature_hex` | Spending signature. |
| `spending_witness` | Witness data. |
| `lifespan` | Time between creation and spending. |
| `cdd` | Coin Days Destroyed. |

---

### 5. **Inputs**
Inputs reference previous outputs and represent the source of funds for a transaction. Each input includes details about the previous output being spent, such as its address, value, and unlocking script.

| Column | Description |
|--------|-------------|
| `block_id` | Block of input. |
| `transaction_hash` | Input's transaction. |
| `index` | Input index. |
| `time` | Input timestamp. |
| `value`, `value_usd` | Input value. |
| `recipient` | Source address. |
| `type` | Script type. |
| `script_hex` | Script in hex. |
| `is_from_coinbase` | From coinbase. |
| `is_spendable` | Spendable flag. |

---

## Architecture

<img width="1536" height="1024" alt="architecture" src="https://github.com/user-attachments/assets/6ace40b7-674e-4599-b7fd-f38edfdec115" />

The project employs a **modern data stack**—including Snowflake, dbt, Airflow, machine learning, and Power BI—to automate data processing, feature engineering, model training, and visualization.
1. **Blockchair (Extract)**  
    - Utilizes Blockchair’s public blockchain data dumps to extract comprehensive datasets for cryptocurrencies like Litecoin.
    - Data includes addresses, blocks, transactions, inputs, and outputs, ensuring a granular view of blockchain activity.

2. **Snowflake (Staging & Storage)**  
    - Acts as a centralized cloud data warehouse for both raw and processed blockchain data.
    - Supports scalable storage and fast querying, enabling efficient data exploration and downstream processing.

3. **dbt (Transformation & Feature Engineering)**  
    - Implements modular SQL models to clean, join, and transform raw blockchain data.
    - Engineers features such as transaction frequency, average balance, and address activity patterns, which are critical for wallet classification.

4. **Dimensional Modeling**  
    - Organizes data into fact tables (e.g., transactions, balances) and dimension tables (e.g., wallet, address, time, exchange).
    - Enables analytical queries and supports business intelligence use cases by providing a structured schema.

5. **Machine Learning (Classification)**  
    - Trains classification models using engineered features to categorize wallets (e.g., personal, exchange, miner, suspicious).
    - Evaluates model performance and iteratively improves feature sets for better accuracy.

6. **Airflow (Orchestration & Automation)**  
    - Schedules and automates the entire data pipeline, including extraction, transformation, model training, and evaluation.
    - Ensures data freshness and reproducibility by managing dependencies and workflow execution.

7. **Power BI (Visualization & Insights)**  
    - Connects to Snowflake to visualize wallet classifications, suspicious activity, and model metrics.
    - Delivers interactive dashboards for stakeholders to monitor trends and investigate anomalies.
---
## Model Training
### 1. Wallet Categories (Target Classes)

- **Small/Individual Wallets:** Low balance and low transaction activity.
- **Whale Wallets:** Very high balance, moderate activity.
- **Exchange/Merchant Wallets:** Very high transaction counts and many counterparties.

### 2. Labeling Strategy 

Since the dataset lacks labeled wallet categories, we assign labels using simple rules based on features like balance and transaction count. This enables supervised model training.

**Label Assignment Rules:**
- **Whale:** Wallets in the top 5% by balance.
- **Exchange/Merchant:** Wallets in the top 5% by transaction count.
- **Small/Individual:** All other wallets.
### 3. Machine Learning Algorithms


We will use **XGBoost** for its high accuracy with tabular data, effective handling of class imbalance, and clear feature importance outputs, alongside **Random Forest** as a strong baseline due to its robustness and ability to mitigate overfitting.





---

| Feature                 | Source File                                            | Direct / Derived | Notes                                                                                       |
| ----------------------- | ------------------------------------------------------ | ---------------- | ------------------------------------------------------------------------------------------- |
| `wallet_address`        | `addresses.tsv`                                        | Direct           | Primary key identifying the wallet                                                          |
| `balance`               | `addresses.tsv`                                        | Direct           | Current balance of the wallet                                                               |
| `first_seen`            | `addresses.tsv`                                        | Direct           | Date the wallet was first observed                                                          |
| `last_seen`             | `addresses.tsv`                                        | Direct           | Date the wallet was last active                                                             |
| `activity_days`         | `addresses.tsv`                                        | Derived          | `last_seen – first_seen`                                                                    |
| `num_incoming_tx`       | `outputs.tsv`                             | Derived          | Count of transactions where wallet received coins                                           |
| `total_received`        | `outputs.tsv`                             | Derived          | Sum of all incoming transaction values                                                      |
| `avg_incoming_value`    | `outputs.tsv`                             | Derived          | Average value of incoming transactions                                                      |
| `num_outgoing_tx`       | `inputs.tsv`                              | Derived          | Count of transactions where wallet sent coins                                               |
| `total_sent`            | `inputs.tsv`                              | Derived          | Sum of all outgoing transaction values                                                      |
| `avg_outgoing_value`    | `inputs.tsv`                              | Derived          | Average value of outgoing transactions                                                      |
| `total_tx`              | Computed                                               | Derived          | `num_incoming_tx + num_outgoing_tx`                                                         |
| `avg_tx_value`          | Computed                                               | Derived          | `(avg_incoming_value + avg_outgoing_value)/2`                                               |
| `inputs_per_tx_avg`     | `transactions.tsv` + `inputs.tsv`         | Derived          | Average number of inputs per outgoing transaction                                           |
| `outputs_per_tx_avg`    | `transactions.tsv` + `outputs.tsv`        | Derived          | Average number of outputs per outgoing transaction                                          |
| `unique_counterparties` | `inputs.tsv` + `outputs.tsv` | Derived          | Number of distinct wallets this wallet interacted with                                      |
| `fee_ratio`             | `transactions.tsv` + `inputs.tsv`         | Derived          | Total fee divided by total transaction value                                   |
| `wallet_category`       | Computed                                               | Derived          | Target variable: Small / Whale / Exchange/Merchant |

---

## Future Enhancements
- **Expand blockchain coverage:** Integrate additional blockchains (e.g., Ethereum, Bitcoin) to broaden analysis and support cross-chain wallet classification.
- **Real-time classification:** Enable streaming data ingestion and near real-time wallet classification for proactive monitoring and alerting.







