"""
einstein.simulation — Numerical field simulations.
"""

from .scalar_field import (
    SimulationResult,
    simulate_kg_1d,
    simulate_kg_2d,
    simulate_phi4_1d,
    simulate_kg_curved_1d,
)

__all__ = [
    "SimulationResult",
    "simulate_kg_1d",
    "simulate_kg_2d",
    "simulate_phi4_1d",
    "simulate_kg_curved_1d",
]
