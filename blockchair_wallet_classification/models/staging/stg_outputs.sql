{{ config(materialized="view") }}

select
    TRANSACTION_HASH as transaction_hash,
    RECIPIENT as address,
    VALUE as output_value_ltc,
    null::date as tx_date
from {{ source("raw", "outputs") }}
