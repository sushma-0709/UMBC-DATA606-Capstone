{{ config(materialized="table") }}

with agg as (
    select
        address,
        count(*) as active_days,
        sum(tx_count) as total_tx_count,
        sum(incoming_tx_count) as total_incoming_tx,
        sum(outgoing_tx_count) as total_outgoing_tx,
        sum(total_received) as lifetime_received_ltc,
        sum(total_sent) as lifetime_sent_ltc,
        sum(daily_fee_ltc) as total_fees_paid_ltc,
        max(balance_ltc) as current_balance_ltc,
        avg(tx_count) as avg_tx_per_day,
        avg(total_received) as avg_received_per_day,
        avg(total_sent) as avg_sent_per_day
    from {{ ref("fct_wallet_daily_features") }}
    group by address
)

select
    *,
    case
        when lifetime_received_ltc > 1000 then 'Whale/Exchange'
        when lifetime_received_ltc between 10 and 1000 then 'Active Wallet'
        else 'Dormant/Small Wallet'
    end as wallet_label
from agg
