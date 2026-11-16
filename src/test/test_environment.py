import pytest
from dataclasses import replace
import numpy as np
from pathlib import Path
from unittest.mock import patch, MagicMock
import sys

# Add the project root directory to Python path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from src.simulation.environment import SimulationEnvironment
from src.config.simulation_config import RewardParameters
from src.utils.simulation_utils import initialize_environment, initialize_agent
from src.test.utils import (
    get_unique_connection_name,
    simulation_config,
    mock_signal,
    mock_traci,
    mock_get_traci,
    mock_sumo_env,
    mock_normalize_traffic_state,
    mock_rl_reward
    )


@patch('src.simulation.environment.SumoEnvironment')
@patch('src.utils.simulation_utils.get_traci')
def test_simulation_environment_initialization(mock_get_traci, mock_sumo_env_class, simulation_config):
    """Test that SimulationEnvironment initializes correctly."""
    connection_name = get_unique_connection_name("test_connection")

    # Set up the mocks
    mock_sumo_env = MagicMock()
    mock_sumo_env_class.return_value = mock_sumo_env

    mock_traci = MagicMock()
    mock_get_traci.return_value = mock_traci

    env = SimulationEnvironment(
        network_config=simulation_config.project_root / simulation_config.map_config["net"],
        map_name=simulation_config.map_name,
        gui=simulation_config.display_gui,
        start_time=0,
        end_time=3600,
        step_length=simulation_config.map_config["step_length"],
        yellow_length=simulation_config.map_config["yellow_length"],
        step_ratio=simulation_config.map_config["step_ratio"],
        max_distance=simulation_config.max_observation_distance,
        use_libsumo=simulation_config.use_libsumo,
        seed=simulation_config.random_seed,
        connection_name=connection_name,
        scale=1.0,
        signal_config=simulation_config.signal_config
    )

    assert isinstance(env, SimulationEnvironment)
    assert env.map_name == simulation_config.map_name
    assert env.start_time == 0
    assert env.end_time == 3600
    assert env.scale == 1.0
    assert env.connection_name == connection_name
    assert env.seed == simulation_config.random_seed
    assert env.use_libsumo == simulation_config.use_libsumo

    env.close()

def test_simulation_environment_reset(simulation_config, mock_sumo_env, mock_normalize_traffic_state, mock_signal, mock_rl_reward):
    """Test environment reset functionality."""
    connection_name = get_unique_connection_name("test_connection_reset")

    # Create environment with mocked components
    env = SimulationEnvironment(
        network_config=simulation_config.project_root / simulation_config.map_config["net"],
        map_name=simulation_config.map_name,
        gui=simulation_config.display_gui,
        start_time=0,
        end_time=3600,
        step_length=simulation_config.map_config["step_length"],
        yellow_length=simulation_config.map_config["yellow_length"],
        step_ratio=simulation_config.map_config["step_ratio"],
        max_distance=simulation_config.max_observation_distance,
        use_libsumo=simulation_config.use_libsumo,
        seed=simulation_config.random_seed,
        connection_name=connection_name,
        scale=1.0,
        signal_config=simulation_config.signal_config,
        reward_function=mock_rl_reward,
        state_function=mock_normalize_traffic_state,
    )

    # Set the state_function to our mock
    env.state_function = mock_normalize_traffic_state

    # Mock the _initialize_signals method to set up our signals
    def mock_initialize_signals(use_all_signals=False):
        env.signals = {"1": mock_signal, "2": mock_signal}
        # Call observe on each signal to simulate the real behavior
        for signal in env.signals.values():
            signal.observe()
        return None

    # Replace the _initialize_signals method with our mock
    env._initialize_signals = mock_initialize_signals

    # Ensure sumo_env is properly mocked
    env.sumo_env = mock_sumo_env

    # Call reset and verify the returned state
    initial_state = env.reset()

    # Verify the state is properly normalized using our mock
    assert isinstance(initial_state, dict)
    assert "1" in initial_state
    assert "2" in initial_state
    assert isinstance(initial_state["1"], np.ndarray)
    assert isinstance(initial_state["2"], np.ndarray)

    # Verify simulation state
    assert not env.is_simulation_complete

    # Verify that reset was called on the SUMO environment
    assert mock_sumo_env.reset.called

    # Verify that signals were initialized
    assert len(env.signals) == 2
    assert "1" in env.signals
    assert "2" in env.signals

    # Verify that the mock signal's observe method was called
    mock_signal.observe.assert_called()

    # Verify that the state function (normalize_traffic_state) was called
    mock_normalize_traffic_state.assert_called()

    env.close()

def test_simulation_environment_step(simulation_config, mock_sumo_env, mock_signal, mock_rl_reward):
    """Test environment step functionality."""
    connection_name = get_unique_connection_name("test_connection_step")

    # Create a SimulationEnvironment instance with mocked reward_function
    env = SimulationEnvironment(
        network_config=simulation_config.project_root / simulation_config.map_config["net"],
        map_name=simulation_config.map_name,
        gui=simulation_config.display_gui,
        start_time=0,
        end_time=3600,
        step_length=simulation_config.map_config["step_length"],
        yellow_length=simulation_config.map_config["yellow_length"],
        step_ratio=simulation_config.map_config["step_ratio"],
        max_distance=simulation_config.max_observation_distance,
        use_libsumo=simulation_config.use_libsumo,
        seed=simulation_config.random_seed,
        connection_name=connection_name,
        scale=1.0,
        signal_config=simulation_config.signal_config,
        reward_function=mock_rl_reward
    )

    # Mock the _initialize_signals method to set up our signals
    def mock_initialize_signals(use_all_signals=False):
        # Create a dictionary with all signal IDs from the config
        env.signals = {}
        for signal_id in simulation_config.signal_config.keys():
            env.signals[signal_id] = mock_signal
        return None

    # Replace the _initialize_signals method with our mock
    env._initialize_signals = mock_initialize_signals

    # Ensure sumo_env is properly mocked
    mock_sumo_env.get_time.return_value = 100  # Less than end_time (3600)
    mock_sumo_env.active_signal_ids = list(simulation_config.signal_config.keys())
    env.sumo_env = mock_sumo_env

    # Call reset to initialize the environment
    env.reset()

    # Create actions for all signals
    actions = {}
    for signal_id in env.signals.keys():
        actions[signal_id] = 0  # Default action for all signals

    # Call step and verify the returned values
    observations, rewards = env.step(actions)

    # Verify that the observations and rewards are dictionaries
    assert isinstance(observations, dict)
    assert isinstance(rewards, dict)

    # Verify that the rewards match what our mock returns
    assert rewards == mock_rl_reward.return_value

    # Verify that the mock signal's methods were called
    mock_signal.prep_phase.assert_called()
    mock_signal.set_phase.assert_called()
    mock_signal.observe.assert_called()

    env.close()

@patch('src.utils.simulation_utils.get_traci')
def test_initialize_environment(mock_get_traci, simulation_config):
    """Test that initialize_environment correctly sets up the environment."""
    mock_traci = MagicMock()
    mock_get_traci.return_value = mock_traci

    connection_name = get_unique_connection_name("test_connection")
    env = initialize_environment(simulation_config, connection_name=connection_name)

    assert env is not None
    assert env.connection_name == connection_name
    assert env.use_libsumo == simulation_config.use_libsumo
    assert env.gui == simulation_config.display_gui
    assert env.step_length == simulation_config.map_config["step_length"]
    assert env.yellow_length == simulation_config.map_config["yellow_length"]
    assert env.step_ratio == simulation_config.map_config["step_ratio"]
    assert env.max_distance == simulation_config.max_observation_distance
    assert env.seed == simulation_config.random_seed

@patch('src.utils.simulation_utils.PPOAgentController')
def test_initialize_rl_agent(mock_ppo_agent, simulation_config):
    """Test that initialize_agent correctly sets up an RL agent."""
    mock_agent = MagicMock()
    mock_ppo_agent.return_value = mock_agent

    # Create a mock environment
    mock_env = MagicMock()
    mock_env.obs_shape = {"1": (1, 2, 3)}
    mock_env.sumo_env.signal_phases = {"1": ["g", "r"]}

    agent = initialize_agent(simulation_config, environment=mock_env, is_rl_agent=True)

    assert agent == mock_agent
    mock_ppo_agent.assert_called_once()

@patch('src.utils.simulation_utils.FixedIntervalAgentController')
def test_initialize_non_rl_agent(mock_fixed_agent, simulation_config):
    """Test that initialize_agent correctly sets up a non-RL agent."""
    mock_agent = MagicMock()
    mock_fixed_agent.return_value = mock_agent

    # Create a mock environment
    mock_env = MagicMock()

    agent = initialize_agent(simulation_config, environment=mock_env, is_rl_agent=False)

    assert agent == mock_agent
    mock_fixed_agent.assert_called_once()

@patch('src.utils.simulation_utils.PPOAgentController')
def test_initialize_agent_with_model(mock_ppo_agent, simulation_config, tmp_path):
    """Test that initialize_agent correctly loads a model when provided."""
    mock_agent = MagicMock()
    mock_ppo_agent.return_value = mock_agent

    # Create a mock environment
    mock_env = MagicMock()
    mock_env.obs_shape = {"1": (1, 2, 3)}
    mock_env.sumo_env.signal_phases = {"1": ["g", "r"]}

    # Create a dummy model file
    model_path = tmp_path / "model.pt"
    model_path.touch()

    agent = initialize_agent(simulation_config, environment=mock_env, is_rl_agent=True, model_dir=model_path)

    assert agent == mock_agent
    mock_ppo_agent.assert_called_once()

@patch('src.simulation.environment.SumoEnvironment')
def test_environment_metrics(mock_sumo_env_class, simulation_config, mock_signal, mock_rl_reward):
    """Test environment metrics calculation."""
    connection_name = get_unique_connection_name("test_connection_metrics")

    # Set up the mock
    mock_sumo_env = MagicMock()
    mock_sumo_env.get_time.return_value = 100  # Return a value less than end_time (3600)
    mock_sumo_env_class.return_value = mock_sumo_env

    env = SimulationEnvironment(
        network_config=simulation_config.project_root / simulation_config.map_config["net"],
        map_name=simulation_config.map_name,
        gui=simulation_config.display_gui,
        start_time=0,
        end_time=3600,
        step_length=simulation_config.map_config["step_length"],
        yellow_length=simulation_config.map_config["yellow_length"],
        step_ratio=simulation_config.map_config["step_ratio"],
        max_distance=simulation_config.max_observation_distance,
        use_libsumo=simulation_config.use_libsumo,
        seed=simulation_config.random_seed,
        connection_name=connection_name,
        scale=1.0,
        signal_config=simulation_config.signal_config,
        reward_function=mock_rl_reward
    )

    # Mock the _initialize_signals method to set up our signals
    def mock_initialize_signals(use_all_signals=False):
        # Create a dictionary with all signal IDs from the config
        env.signals = {}
        for signal_id in simulation_config.signal_config.keys():
            env.signals[signal_id] = mock_signal
        return None

    # Replace the _initialize_signals method with our mock
    env._initialize_signals = mock_initialize_signals

    # Ensure sumo_env is properly mocked
    env.sumo_env = mock_sumo_env

    # Call reset to initialize the environment
    env.reset()

    # Create actions for all signals
    actions = {}
    for signal_id in env.signals.keys():
        actions[signal_id] = 0  # Default action for all signals

    # Call step to update metrics
    env.step(actions)

    # Get metrics and verify they exist
    metrics = env.get_metrics()
    assert metrics is not None

    # Verify that the metrics have the expected attributes
    assert hasattr(metrics, 'step')
    assert hasattr(metrics, 'rewards')
    assert hasattr(metrics, 'waiting_times')
    assert hasattr(metrics, 'queue_lengths')
    assert hasattr(metrics, 'co2_emissions')
    assert hasattr(metrics, 'avg_speeds')
    assert hasattr(metrics, 'fuel_consumption')
    assert hasattr(metrics, 'noise_emission')

    env.close()


@patch.object(SimulationEnvironment, "__post_init__", return_value=None)
def test_configured_reward_weights_reach_environment_step(_initialize, simulation_config):
    configured = replace(simulation_config, reward_params=RewardParameters(queue_penalty=3))
    lane = {"total_wait": 0, "queue": 2, "approach": 0, "vehicles": []}
    signal = MagicMock(lanes=["lane"], phase=0, full_observation={"lane": lane})

    for config, expected in ((simulation_config, -2), (configured, -6), (simulation_config, -2)):
        env = initialize_environment(config, "test_reward", start_time=0, end_time=100)
        env.sumo_env = MagicMock(active_signal_ids=["signal"])
        env.sumo_env.get_time.return_value = 10
        env.signals = {"signal": signal}

        _, rewards = env.step({"signal": 0})

        assert rewards == {"signal": expected}
