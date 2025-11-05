{{ config(materialized="view") }}

select
    ID as block_id,
    TIME::date as block_date,
    SIZE as size,
    TRANSACTION_COUNT as transaction_count,
    REWARD as reward_ltc,
    GUESSED_MINER as guessed_miner
from {{ source("raw", "blocks") }}
