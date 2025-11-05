{{ config(materialized='view') }}

with tx as (
    select transaction_hash, tx_date
    from {{ ref("stg_transactions") }}
),

incoming as (
    select
        o.transaction_hash,
        t.tx_date,
        o.address,
        o.output_value_ltc as value_ltc,
        'incoming' as direction
    from {{ ref("stg_outputs") }} o
    join tx t using (transaction_hash)
),

outgoing as (
    select
        i.transaction_hash,
        t.tx_date,
        i.address,
        i.input_value_ltc as value_ltc,
        'outgoing' as direction
    from {{ ref("stg_inputs") }} i
    join tx t using (transaction_hash)
)

select * from incoming
union all
select * from outgoing
