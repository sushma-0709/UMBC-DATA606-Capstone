{{ config(materialized="view") }}

select
    ADDRESS as address,
    BALANCE as balance_ltc
from {{ source("raw", "addresses") }}