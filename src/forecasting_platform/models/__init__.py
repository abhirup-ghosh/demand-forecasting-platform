"""Model tiers. Every tier returns a long-format frame with exactly ``FORECAST_COLUMNS``."""

FORECAST_COLUMNS = ["unique_id", "ds", "model_name", "yhat", "yhat_lo80", "yhat_hi80"]
