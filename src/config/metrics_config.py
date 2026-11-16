from dataclasses import dataclass
from typing import Dict


@dataclass
class Metrics:

    step: float
    rewards: float
    waiting_times: Dict[str, float]
    queue_lengths: Dict[str, int]
    co2_emissions: Dict[str, float]
    avg_speeds: Dict[str, float]
    fuel_consumption: Dict[str, float]
    noise_emission: Dict[str, float]
