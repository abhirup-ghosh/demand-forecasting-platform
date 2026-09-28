"""Asymmetric business cost of forecast error (PLAN.md P0.7).

``cost = sum(cost_under * max(0, actual - forecast) + cost_over * max(0, forecast - actual))``

The default **3:1 under:over ratio is illustrative**, not fitted from Favorita's real financials:
for perishable grocery a unit of under-forecast (a stock-out / lost sale) is assumed to cost more
than a unit of over-forecast (holding cost / waste). The "right" ratio is a business input, not a
data-science one — which is why it is an adjustable slider in the dashboard (P0.10) rather than a
fixed constant. Units are "cost per unit of sales", so totals compare models, not currencies.
"""

import numpy as np

from forecasting_platform.config import settings


def business_cost(
    y_true,
    y_pred,
    cost_under: float = settings.COST_UNDER_PER_UNIT,
    cost_over: float = settings.COST_OVER_PER_UNIT,
) -> float:
    y, p = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)
    under = np.clip(y - p, 0, None)
    over = np.clip(p - y, 0, None)
    return float(np.sum(cost_under * under + cost_over * over))
