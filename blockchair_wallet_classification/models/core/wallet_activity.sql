{{ config(materialized='table') }}

with sent as (
    select address, sum(input_value_ltc) as total_sent
    from {{ ref('stg_inputs') }}
    group by 1
),
received as (
    select address, sum(output_value_ltc) as total_received
    from {{ ref('stg_outputs') }}
    group by 1
),
balances as (
    select
        coalesce(s.address, r.address) as address,
        coalesce(total_received, 0) as total_received,
        coalesce(total_sent, 0) as total_sent,
        coalesce(total_received, 0) - coalesce(total_sent, 0) as net_balance
    from sent s
    full outer join received r on s.address = r.address
)
select * from balances
