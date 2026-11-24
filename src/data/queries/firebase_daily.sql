-- Firebase cohorts for one geo/platform, pivoted from long (age x type rows) to wide:
-- one row per install date. This "conditional aggregation" pattern is a standard SQL pivot.
SELECT
  install_date,
  date_week,
  SUM(IF(type = 'other'        AND age = 0, users,  0)) AS installs_fb,   -- installs = D0 active users
  SUM(IF(type = 'other'        AND age = 1, users,  0)) AS d1_actives,
  SUM(IF(type = 'ad_revenue'   AND age = 0, value,  0)) AS ad_rev_d0,
  SUM(IF(type = 'ad_revenue'   AND age = 1, value,  0)) AS ad_rev_d1,
  SUM(IF(type = 'Banner'       AND age = 0, events, 0)) AS imps_d0_banner,
  SUM(IF(type = 'Interstitial' AND age = 0, events, 0)) AS imps_d0_interstitial,
  SUM(IF(type = 'Rewarded'     AND age = 0, events, 0)) AS imps_d0_rewarded,
  SUM(IF(type = 'Banner'       AND age = 1, events, 0)) AS imps_d1_banner,
  SUM(IF(type = 'Interstitial' AND age = 1, events, 0)) AS imps_d1_interstitial,
  SUM(IF(type = 'Rewarded'     AND age = 1, events, 0)) AS imps_d1_rewarded
FROM `{table}`
WHERE country = @country
  AND platform = @platform
  AND date_week BETWEEN @start_week AND @end_week
  AND age <= 1
GROUP BY install_date, date_week
ORDER BY install_date
