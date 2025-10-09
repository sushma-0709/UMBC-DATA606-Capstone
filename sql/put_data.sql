-- Upload Litecoin data to Snowflake stage

-- Addresses
PUT file:///root/UMBC-DATA606-Capstone/data/litecoin/addresses/blockchair_litecoin_addresses_*.tsv.gz
@LITECOIN_STAGE
AUTO_COMPRESS=TRUE
OVERWRITE=TRUE;

-- Blocks
PUT file:///root/UMBC-DATA606-Capstone/data/litecoin/blocks/blockchair_litecoin_blocks_*.tsv.gz
@LITECOIN_STAGE
AUTO_COMPRESS=TRUE;

-- Inputs
PUT file:///root/UMBC-DATA606-Capstone/data/litecoin/inputs/blockchair_litecoin_inputs_*.tsv.gz
@LITECOIN_STAGE
AUTO_COMPRESS=TRUE;

-- Outputs
PUT file:///root/UMBC-DATA606-Capstone/data/litecoin/outputs/blockchair_litecoin_outputs_*.tsv.gz
@LITECOIN_STAGE
AUTO_COMPRESS=TRUE;

-- Transactions
PUT file:///root/UMBC-DATA606-Capstone/data/litecoin/transactions/blockchair_litecoin_transactions_*.tsv.gz
@LITECOIN_STAGE
AUTO_COMPRESS=TRUE;
