{{ config(materialized="table") }}

with base as (
    select
        address,
        tx_date,
        sum(case when direction = 'incoming' then value_ltc else 0 end) as total_received,
        sum(case when direction = 'outgoing' then value_ltc else 0 end) as total_sent,
        count(*) as tx_count,
        sum(case when direction = 'incoming' then 1 else 0 end) as incoming_tx_count,
        sum(case when direction = 'outgoing' then 1 else 0 end) as outgoing_tx_count
    from {{ ref("int_wallet_transfers") }}
    group by address, tx_date
),

fees as (
    select
        address,
        tx_date,
        sum(fee_ltc) as daily_fee_ltc
    from {{ ref("stg_transactions") }} t
    join {{ ref("stg_inputs") }} i using (transaction_hash)
    group by address, tx_date
),

final as (
    select
        b.address,
        b.tx_date,
        b.tx_count,
        b.incoming_tx_count,
        b.outgoing_tx_count,
        b.total_received,
        b.total_sent,
        coalesce(f.daily_fee_ltc,0) as daily_fee_ltc,
        a.balance_ltc,
        case when b.tx_count > 0 then 1 else 0 end as active_flag
    from base b
    left join fees f on b.address = f.address and b.tx_date = f.tx_date
    left join {{ ref("stg_addresses") }} a on b.address = a.address
)

select * from final
