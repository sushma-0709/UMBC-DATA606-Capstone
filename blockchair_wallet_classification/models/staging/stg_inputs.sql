{{ config(materialized="view") }}

select
    TRANSACTION_HASH as transaction_hash,
    RECIPIENT as address,
    VALUE as input_value_ltc,
    null::date as tx_date -- date will come from transaction table in join
from {{ source("raw", "inputs") }}
