from dataclasses import dataclass, field
from pathlib import Path
from .data_config import map_configs, signal_configs
from typing import Dict

from .path_config import RESULTS_DIR, ROOT, PRETRAINED_MODELS_DIR
from src.utils.simulation_utils import set_random_seeds

# Configuration dictionary
SIM_CONFIG = {
    "display_gui": False,
    "load_model": False,
    "load_pretrained_model": False,
    "training_episodes": 3,
    "map_name": "synthetic-cross",
    "max_observation_distance": 200,
    "random_seed": 7,
    "use_libsumo": False,
    "hyperparameters": {
        "learning_rate": 2.5e-3,
        "optimizer_epsilon": 1e-5,
        "policy_clip_range": 0.1,
        "value_clip_range": None,
        "update_interval": 256,
        "minibatch_size": 128,
        "ppo_epochs": 6,
        "standardize_advantages": True,
        "entropy_coefficient": 0.001,
        "max_gradient_norm": 0.5
    },
    "reward_params": {
        "waiting_time_penalty": 1,
        "throughput_reward": 1,
        "journey_reward": 1,
        "queue_penalty": 1
    }
}


@dataclass(frozen=True)
class RewardParameters:
    waiting_time_penalty: int = SIM_CONFIG["reward_params"]["waiting_time_penalty"]
    throughput_reward: int = SIM_CONFIG["reward_params"]["throughput_reward"]
    journey_reward: int = SIM_CONFIG["reward_params"]["journey_reward"]
    queue_penalty: int = SIM_CONFIG["reward_params"]["queue_penalty"]

@dataclass(frozen=True)
class HyperParameters:
    learning_rate: float = SIM_CONFIG["hyperparameters"]["learning_rate"]
    optimizer_epsilon: float = SIM_CONFIG["hyperparameters"]["optimizer_epsilon"]
    policy_clip_range: float = SIM_CONFIG["hyperparameters"]["policy_clip_range"]
    value_clip_range: float = SIM_CONFIG["hyperparameters"]["value_clip_range"]
    update_interval: int = SIM_CONFIG["hyperparameters"]["update_interval"]
    minibatch_size: int = SIM_CONFIG["hyperparameters"]["minibatch_size"]
    ppo_epochs: int = SIM_CONFIG["hyperparameters"]["ppo_epochs"]
    standardize_advantages: bool = SIM_CONFIG["hyperparameters"]["standardize_advantages"]
    entropy_coefficient: float = SIM_CONFIG["hyperparameters"]["entropy_coefficient"]
    max_gradient_norm: float = SIM_CONFIG["hyperparameters"]["max_gradient_norm"]

@dataclass(frozen=True)
class SimulationConfig:
    # Base paths
    project_root: Path = ROOT
    results_directory: Path = RESULTS_DIR
    pretrained_models_directory: Path = PRETRAINED_MODELS_DIR / SIM_CONFIG["map_name"]

    # General settings
    display_gui: bool = SIM_CONFIG["display_gui"]
    load_model: bool = SIM_CONFIG["load_model"]
    load_pretrained_model: bool = SIM_CONFIG["load_pretrained_model"]
    training_episodes: int = SIM_CONFIG["training_episodes"]
    map_name: str = SIM_CONFIG["map_name"]
    max_observation_distance: int = SIM_CONFIG["max_observation_distance"]
    random_seed: int = SIM_CONFIG["random_seed"]
    use_libsumo: bool = SIM_CONFIG["use_libsumo"] if not SIM_CONFIG["display_gui"] else False

    # Specialized configurations
    hyperparameters: HyperParameters = HyperParameters()
    reward_params: RewardParameters = RewardParameters()
    map_config: Dict = field(default_factory=lambda: map_configs[SIM_CONFIG["map_name"]])
    signal_config: Dict = field(default_factory=lambda: signal_configs[SIM_CONFIG["map_name"]])

    def __post_init__(self):
        self._set_random_seeds()

    def _set_random_seeds(self):
        # Set random seeds for reproducibility
        set_random_seeds(self.random_seed)
