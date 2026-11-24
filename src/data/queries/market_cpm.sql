-- CPM per ad type for EVERY geo/platform, for two weeks. Lets the agent check whether a CPM move
-- is market-wide (e.g. Q4 seasonality) or specific to the market being analysed.
SELECT
  date_week,
  country,
  platform,
  type,
  SUM(impressions) AS impressions,
  SUM(revenue)     AS revenue
FROM `{table}`
WHERE date_week IN (@prev_week, @week)
GROUP BY date_week, country, platform, type
