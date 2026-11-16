"""
This module provides the primary environment interface between SUMO traffic simulation
and reinforcement learning agents. It manages multiple traffic signals, observation
collection, action execution, and reward calculation for traffic optimization.
"""

import gym
import numpy as np

from pathlib import Path
from .signal_controller import Signal
from dataclasses import dataclass, field
from src.config.metrics_config import Metrics
from typing import Dict, List, Callable
from src.components.states import normalize_traffic_state
from src.config.environment_config import SumoEnvironment
from src.components.rewards import rl_reward, calculate_wait_times
from src.utils.simulation_utils import calculate_emission_metric

@dataclass
class SimulationEnvironment(gym.Env):
    """
    Environment for controlling multiple traffic signals using reinforcement learning.
    """

    network_config: Path = Path("")
    map_name: str = ""
    gui: bool = False
    start_time: int = -1
    end_time: int = -1
    step_length: int = -1
    yellow_length: int = -1
    step_ratio: int = -1
    max_distance: int = -1
    use_libsumo: bool = False
    seed: int = 7
    connection_name: str = ""
    scale: float = -1.0
    signal_config: Dict = field(default_factory=dict)

    state_function: Callable = normalize_traffic_state
    reward_function: Callable = rl_reward

    is_simulation_complete: bool = False
    sumo_env: SumoEnvironment = None

    wait_metric: Dict[str, float] = field(default_factory=dict, init=False)
    signals: Dict[str, Signal] = field(default_factory=dict, init=False)
    observation_space: List = field(default_factory=list, init=False)
    action_space: List = field(default_factory=list, init=False)
    obs_shape: Dict = field(default_factory=dict, init=False)

    _rewards: Dict[str, float] = field(default_factory=dict, init=False)

    def __post_init__(self):
        """
        Sets up the SUMO simulation for training/testing.
        """
        # Start SUMO
        self.sumo_env = SumoEnvironment(
            network_config=self.network_config,
            seed=self.seed,
            start_time=self.start_time,
            end_time=self.end_time,
            connection_name=self.connection_name,
            gui=self.gui,
            libsumo=self.use_libsumo,
            step_ratio=self.step_ratio,
            scale=self.scale,
        )
        self._initialize_signals(use_all_signals=True)
        self._setup_observation_action_spaces()
        self.sumo_env.close()

        print(f"Environment initialized for map: {self.map_name}")

    def _initialize_signals(self, use_all_signals=False):
        """
        Initialize Signal objects for traffic signals.
        w"""
        # Determine which signal IDs to use
        signal_ids = self.sumo_env.signal_ids if use_all_signals else self.sumo_env.active_signal_ids

        # Initialize Signal objects
        for signal_id in signal_ids:
            self.signals[signal_id] = Signal(
                signal_config=self.signal_config,
                sumo=self.sumo_env.connection,
                signal_id=signal_id,
                yellow_light_duration=self.yellow_length,
                phases=self.sumo_env.signal_phases[signal_id],
            )

            # Only initialize wait metrics when setting up active signals
            if not use_all_signals:
                self.wait_metric[signal_id] = 0.0

        # Connect signals to each other and collect initial observations
        for signal_id in signal_ids:
            self.signals[signal_id].signals = self.signals
            self.signals[signal_id].observe(self.step_length, self.max_distance)

        # Remove temporary variables
        del signal_ids

    def _setup_observation_action_spaces(self):
        """
        Define observation and action spaces
        """

        observations = self.state_function(self.signals)
        self.obs_shape = {signal_id: observations[signal_id].shape for signal_id in observations}

        # Create Gym spaces for each signal
        self.observation_space = [
            gym.spaces.Box(low=-np.inf, high=np.inf, shape=observations[signal_id].shape)
            for signal_id in observations
            if signal_id != "top_mgr" and signal_id != "bot_mgr"
        ]
        self.action_space = [
            gym.spaces.Discrete(len(self.sumo_env.signal_phases[signal_id]))
            for signal_id in observations
            if signal_id != "top_mgr" and signal_id != "bot_mgr"
        ]

    @property
    def rewards(self):
        return self._rewards # Get the current rewards for all traffic signals

    def reset(self):
        """
        Reset the environment for a new episode
        """
        self.sumo_env.reset()
        self.np_random, _ = gym.utils.seeding.np_random(self.seed)
        self._initialize_signals(use_all_signals=False)  # Initialize only active signals
        self.is_simulation_complete = False

        return self.state_function(self.signals)

    def step(self, actions):
        """
        Execute one environment step by applying actions to all traffic signals
        """

        # Prepare phase transitions for all signals
        for signal_id in self.signals:
            self.signals[signal_id].prep_phase(actions[signal_id])

        # Simulate the yellow light duration
        for _ in range(self.yellow_length):
            self.sumo_env.step()

        # Apply the new phases to all active signals
        for signal_id in self.sumo_env.active_signal_ids:
            self.signals[signal_id].set_phase()

        # Simulate the remainder of the step (green light duration)
        for _ in range(self.step_length - self.yellow_length):
            self.sumo_env.step()

        # Update observations for all active signals
        for signal_id in self.sumo_env.active_signal_ids:
            self.signals[signal_id].observe(self.step_length, self.max_distance)

        # Process observations and calculate rewards
        observations = self.state_function(self.signals)
        self._rewards = self.reward_function(self.signals)

        # Check if simulation has reached its end time
        current_time = self.sumo_env.get_time()
        self.is_simulation_complete = current_time >= self.end_time

        # Remove temporary variables
        del current_time

        return observations,self._rewards

    def get_metrics(self) -> Metrics:
        """
        Calculate and package performance metrics for the current step
        """
        return Metrics(
            step=self.sumo_env.get_time(),
            rewards=self._calculate_rewards(),
            waiting_times=self._calculate_waiting_times(),
            queue_lengths=self._calculate_queue_lengths(),
            co2_emissions=self._calculate_emission_metric("co2"),
            avg_speeds=self._calculate_avg_speeds(),
            fuel_consumption=self._calculate_emission_metric("fuel"),
            noise_emission=self._calculate_emission_metric("noise"),
        )

    def _calculate_metric(self, metric_extractor):
        """
        Generic helper function to calculate metrics for each signal.
        """
        return {signal_id: metric_extractor(signal) for signal_id, signal in self.signals.items()}

    def _calculate_queue_lengths(self) -> dict:
        """
        Calculate queue lengths for each traffic signal
        """
        def extract_queue_length(signal):
            return sum(signal.full_observation[lane]["queue"] for lane in signal.lanes)

        return self._calculate_metric(extract_queue_length)

    def _calculate_emission_metric(self, emission_type) -> dict:
        """
        Generic function to calculate emission-based metrics for each signal
        """
        def extract_emission(signal):
            active_vehicles = set(self.sumo_env.get_vehicle_ids())

            # Get nearby vehicles
            nearby_vehicles = set()
            for lane in signal.lanes:
                for vehicle in signal.full_observation[lane]["vehicles"]:
                    nearby_vehicles.add(vehicle["id"])

            # Filter to only include active vehicles
            nearby_active_vehicles = nearby_vehicles.intersection(active_vehicles)

            # Use utility function to calculate emissions
            return calculate_emission_metric(
                self.sumo_env.connection,
                nearby_active_vehicles,
                emission_type
            )

        return self._calculate_metric(extract_emission)

    def _calculate_avg_speeds(self) -> dict:
        """
        Calculate average speeds for vehicles around each traffic signal
        """
        def extract_avg_speeds(signal):
            speeds = []
            for lane in signal.lanes:
                speeds.extend([vehicle["speed"] for vehicle in signal.full_observation[lane]["vehicles"]])

            if not speeds:
                return 0

            # Convert m/s to km/h
            return (sum(speeds) / len(speeds)) * 3.6

        return self._calculate_metric(extract_avg_speeds)

    def _calculate_waiting_times(self) -> dict:
        return calculate_wait_times(self.signals)

    def _calculate_rewards(self) -> float:
        return sum(self._rewards.values())

    def close(self):
        self.sumo_env.close() # Close the SUMO connection
