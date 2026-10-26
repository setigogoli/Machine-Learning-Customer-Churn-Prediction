"""Compare actions on one consistent expected-loss baseline."""

import numpy as np
import pandas as pd


def retention_scenario(probabilities, loss=500.0, low_cost=50.0,
                       high_cost=100.0, low_success=0.30, high_success=0.60):
    """Choose the least costly action under hypothetical intervention effects.

    Success is the fraction of otherwise-churning customers retained.
    Intervention cost is paid for every targeted customer. No effect is
    estimated from these observational data.
    """
    p = np.asarray(probabilities, dtype=float)
    if p.ndim != 1 or not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError("Probabilities must be a finite 1D array in [0, 1].")
    if not np.isfinite([loss, low_cost, high_cost]).all() or min(loss, low_cost, high_cost) < 0:
        raise ValueError("Loss and costs must be finite and nonnegative.")
    if not (0 <= low_success <= 1 and 0 <= high_success <= 1):
        raise ValueError("Success rates must be in [0, 1].")
    costs = pd.DataFrame({
        "none": p * loss,
        "low": p * (1 - low_success) * loss + low_cost,
        "high": p * (1 - high_success) * loss + high_cost,
    })
    decisions = pd.DataFrame({"calibrated_churn_probability": p})
    decisions["action"] = costs.idxmin(axis=1)
    decisions["expected_loss"] = costs.min(axis=1)
    decisions["expected_savings_vs_none"] = costs["none"] - costs.min(axis=1)
    totals = pd.DataFrame({
        "strategy": ["none", "low_all", "high_all", "targeted"],
        "expected_loss": [costs.none.sum(), costs.low.sum(), costs.high.sum(), costs.min(axis=1).sum()],
    })
    totals["expected_savings_vs_none"] = totals.expected_loss.iloc[0] - totals.expected_loss
    return decisions, totals
