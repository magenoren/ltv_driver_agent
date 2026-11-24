-- Most recent week in which all 7 install dates are present.
SELECT MAX(date_week) AS week
FROM (
  SELECT date_week, COUNT(DISTINCT install_date) AS n_days
  FROM `{table}`
  WHERE country = @country AND platform = @platform
  GROUP BY date_week
)
WHERE n_days = 7
