# Summary

Eligible rows: 8245 of 54294 raw rows.

Positive rate: 36.980%.

Best overall by F1: hist_gradient_boosting on temporal split / strict spec.

Target definition: Approximate label: round_B > 0 and last_funding_at - first_funding_at <= 36 months; kept only if first_funding_at <= 2015-01-01 and (positive or at least 36 months of observation).

Limitation: The CSV does not include the exact date of the Series B event, so the milestone label is approximate.
