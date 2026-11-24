-- MMP cohorts for one geo/platform: one row per install date.
-- {table} is filled in by Python (table names can't be query parameters); everything else is a parameter.
SELECT
  install_date,
  date_week,
  SUM(installs)      AS installs,
  SUM(cohort_rev_d0) AS rev_d0,
  SUM(cohort_rev_d1) AS rev_d1          -- cumulative D0 + D1 revenue
FROM `{table}`
WHERE country = @country
  AND platform = @platform
  AND date_week BETWEEN @start_week AND @end_week
GROUP BY install_date, date_week
ORDER BY install_date
