SELECT
	store_id,
	COUNT(*) as total_customers
FROM
	{{ ref('customer_base') }}
GROUP BY
      1
