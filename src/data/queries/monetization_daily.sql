-- Network revenue for one geo/platform: one row per date x network x ad type (not cohorted).
SELECT
  date,
  date_week,
  network,
  type,
  SUM(impressions) AS impressions,
  SUM(revenue)     AS revenue
FROM `{table}`
WHERE country = @country
  AND platform = @platform
  AND date_week BETWEEN @start_week AND @end_week
GROUP BY date, date_week, network, type
ORDER BY date, type, network
