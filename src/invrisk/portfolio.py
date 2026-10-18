"""The exposure: commodity inventories held by a euro-area firm, priced in USD.

Risk factors are the log prices of each commodity (USD) and of EUR/USD. The euro value of position i is
    V_i = q_i * S_i / fx,
so a change of the factors x = (log S_1, ..., log S_n, log fx) revalues the book exactly as
    V(x + dx) = sum_i V_i * exp(dS_i - dfx)         (full revaluation),
and to first order dV = sum_i V_i (dS_i - dfx)      (delta approximation).
"""
from dataclasses import dataclass

import numpy as np

TROY_OUNCES_PER_TONNE = 1e6 / 31.1034768


@dataclass
class Inventory:
    quantities: dict          # commodity -> quantity in the price unit (silver: troy oz, copper: t)

    @classmethod
    def course_case(cls, silver_t=500.0, copper_t=40000.0):
        """The course exercise: 500 t of silver (given). Copper: 40 000 t, an assumption (the course gives no
        quantity) chosen so that both positions have a similar euro value (about EUR 300 m each in Nov 2023)."""
        return cls({"silver": silver_t * TROY_OUNCES_PER_TONNE, "copper": copper_t})

    @property
    def names(self):
        return list(self.quantities)

    def values_eur(self, prices):
        """Euro value of each position at the given prices (Series or DataFrame row with the commodity and fx)."""
        return np.array([self.quantities[c] * prices[c] / prices["fx"] for c in self.names], dtype=float)

    def pnl(self, prices, dx):
        """Full-revaluation P&L (EUR) for scenarios dx: array (n_scen, n_commodities + 1) of log-factor changes,
        columns in the order names + ['fx']. Returns (n_scen, n_commodities) P&L per position."""
        v = self.values_eur(prices)
        dx = np.atleast_2d(dx)
        return v * (np.exp(dx[:, :-1] - dx[:, [-1]]) - 1.0)

    def delta(self, prices):
        """Sensitivities of the EUR value to the log factors: dV/dlogS_i = V_i, dV/dlogfx = -sum V_i."""
        v = self.values_eur(prices)
        return np.r_[v, -v.sum()]
