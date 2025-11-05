{{ config(materialized="view") }}

select
    HASH as transaction_hash,
    BLOCK_ID as block_id,
    TIME::date as tx_date,
    INPUT_TOTAL as input_total_ltc,
    OUTPUT_TOTAL as output_total_ltc,
    FEE as fee_ltc
from {{ source("raw", "transactions") }}
