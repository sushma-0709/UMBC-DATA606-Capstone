{{ config(materialized='table') }}

select
    b.block_id,
    b.block_date,
    b.transaction_count as total_transactions,
    sum(t.fee_ltc) as total_fees_ltc,
    avg(t.fee_ltc) as avg_fee_ltc,
    sum(t.output_total_ltc) as total_output_ltc,
    sum(t.input_total_ltc) as total_input_ltc
from {{ ref('stg_blocks') }} b
left join {{ ref('stg_transactions') }} t on b.block_id = t.block_id
group by b.block_id, b.block_date, b.transaction_count
