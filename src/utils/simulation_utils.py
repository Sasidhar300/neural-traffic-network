from typing import Dict, Any
from pathlib import Path
from functools import partial
from traci.exceptions import TraCIException
import numpy as np
import torch
import random
import shutil
import json
import sumolib

def get_traci():
    """
    Returns the appropriate SUMO interface (traci or libsumo)
    """
    from src.config.simulation_config import SIM_CONFIG
    if SIM_CONFIG["use_libsumo"]:
        import libsumo as traci
    else:
        import traci
    return traci

from src.components.agents import PPOAgentController, FixedIntervalAgentController

def initialize_environment(
    simulation_config,
    connection_name: str,
    start_time: int = None,
    end_time: int = None,
    scale: float = 1.0,
):
    """
    Initialize a SUMO environment with appropriate settings.
    """
    from src.simulation.environment import SimulationEnvironment
    from src.components.rewards import rl_reward

    return SimulationEnvironment(
        network_config=simulation_config.project_root / simulation_config.map_config["net"],
        map_name=simulation_config.map_name,
        gui=simulation_config.display_gui,
        start_time=start_time,
        end_time=end_time,
        step_length=simulation_config.map_config["step_length"],
        yellow_length=simulation_config.map_config["yellow_length"],
        step_ratio=simulation_config.map_config["step_ratio"],
        max_distance=simulation_config.max_observation_distance,
        use_libsumo=simulation_config.use_libsumo,
        seed=simulation_config.random_seed,
        connection_name=connection_name,
        scale=scale,
        reward_function=partial(rl_reward, params=simulation_config.reward_params),
        signal_config=simulation_config.signal_config
    )


def initialize_agent(
    simulation_config,
    environment,
    is_rl_agent: bool = True,
    model_dir: Path = None,
) -> Any:
    """
    Initialize the respective agent (RL or non-RL).
    """

    if is_rl_agent:
        # Prepare observation and action space mappings
        observation_action_pairs = {}
        for signal_id in environment.obs_shape:
            if signal_id in environment.sumo_env.signal_phases:
                observation_action_pairs[signal_id] = [
                    environment.obs_shape[signal_id],
                    len(environment.sumo_env.signal_phases[signal_id]),
                ]
            else:
                observation_action_pairs[signal_id] = [environment.obs_shape[signal_id], None]

        # Initialize the PPO Agent Controller
        agent = PPOAgentController(
            sim_config=simulation_config,
            obs_act=observation_action_pairs,
            episodes=simulation_config.training_episodes,
        )

        if model_dir:
            print(f"Loading model from {model_dir}")
            agent.load(Path(model_dir))
            agent.training = False  # Switch to testing only

        return agent

    else:
        # Return baseline non-RL agent
        return FixedIntervalAgentController()


def set_random_seeds(seed: int) -> None:
    """
    Set random seeds for reproducibility across different libraries.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)

    # Hardware specific seeds
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_connection_name(with_model: bool) -> str:
    """
    Return agent type string based on whether a model is being used.
    """
    return "rl" if with_model else "non_rl"


def get_model_dir(training_dir: Path) -> Path:
    """
    Get the path for saving/loading model files.
    """
    return training_dir / "rl_model_weights"


def prepare_directory(directory_path, clean = False):
    """
    Create a directory if it doesn't exist.
    """
    if clean and directory_path.exists():
        shutil.rmtree(directory_path)

    directory_path.mkdir(parents=True, exist_ok=True)
    return directory_path


def save_json(data, file_path):
    """
    Save data to a JSON file
    """
    # Create parent directory if it doesn't exist
    file_path.parent.mkdir(parents=True, exist_ok=True)

    with open(file_path, 'w') as f:
        json.dump(data, f, indent=2)
    return True


def calculate_average(values: Dict[str, float]) -> float:
    """
    Calculate the average value from a dictionary of values.
    """
    if not values:
        return 0.0
    return sum(values.values()) / len(values)


def start_sumo_connection(
    config_path: Path,
    connection_name: str,
    seed: int,
    start_time: int,
    end_time: int,
    scale: float = 1.0,
    use_gui: bool = False,
    libsumo: bool = False,
) -> Any:
    """
    Start a SUMO simulation with the specified parameters.
    """
    # Get the traci interface
    traci = get_traci()

    # Create SUMO command
    if use_gui:
        sumo_binary = [sumolib.checkBinary("sumo-gui"), "--start"]
    else:
        sumo_binary = [sumolib.checkBinary("sumo")]

    # Build command parameters
    common_params = [
        "-c", str(config_path),
        "--no-warnings", "True",
        "--quit-on-end",
        "--seed", str(seed),
        "--begin", str(start_time),
        "--end", str(end_time),
        "--scale", str(scale),
        "--time-to-teleport", "-1",
        "--no-step-log", "True",
        "--collision.check-junctions", "true",
        "--collision.action", "teleport",
    ]

    try:
        traci.start(
            sumo_binary + common_params,
            label=connection_name if not libsumo else "default",
        )
    except TraCIException as e:
        if "already active" in str(e).lower():
            close_sumo_connection(connection_name, libsumo)
            traci.start(
                sumo_binary + common_params,
                label=connection_name if not libsumo else "default",
            )
        else:
            raise



    # Get the appropriate connection
    connection = traci if libsumo else traci.getConnection(connection_name)
    return connection


def calculate_average_metric(values_dict, metric_fn = None):
    """
    Calculate the average of values in a dictionary
    """
    if not values_dict:
        return 0.0

    if metric_fn:
        values = [metric_fn(v) for v in values_dict.values()]
        return sum(values) / len(values)

    return sum(values_dict.values()) / len(values_dict)


def calculate_emission_metric(connection, vehicle_ids, emission_type):
    """
    Calculate emission-based metrics (CO2, fuel, noise) for a set of vehicles.
    """
    if not vehicle_ids:
        return 0.0

    scaling_factors = {
        "co2": 0.001,  # Convert from mg to g
        "fuel": 0.025 * 60 / 1000,  # Convert to L/min
        "noise": 1  # No conversion needed
    }

    values = []
    for vehicle_id in vehicle_ids:
        if emission_type == "co2":
            values.append(connection.vehicle.getCO2Emission(vehicle_id))
        elif emission_type == "fuel":
            values.append(connection.vehicle.getFuelConsumption(vehicle_id))
        elif emission_type == "noise":
            values.append(connection.vehicle.getNoiseEmission(vehicle_id))

    if not values:
        return 0.0

    # Calculate average and apply scaling
    return (sum(values) / len(values)) * scaling_factors.get(emission_type, 1)


def close_sumo_connection(connection_name = None, libsumo = False):
    """
    Close a SUMO connection properly.
    """
    # Get the traci interface
    traci = get_traci()

    if not libsumo and connection_name:
        traci.switch(connection_name)
    traci.close()
