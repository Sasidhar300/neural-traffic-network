import pytest
import numpy as np
from pathlib import Path
from unittest.mock import patch, MagicMock
import sys

# Add the project root directory to Python path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from src.config.simulation_config import SimulationConfig
from src.simulation.signal_controller import Signal
from src.config.environment_config import SumoEnvironment

# Global test counter to ensure unique connection names
test_counter = 0

def get_unique_connection_name(prefix):
    """Generate a unique connection name for each test."""
    global test_counter
    test_counter += 1
    return f"{prefix}_{test_counter}"

@pytest.fixture
def simulation_config():
    """Fixture for simulation configuration."""
    return SimulationConfig()

@pytest.fixture
def test_dir(tmp_path):
    """Fixture for temporary test directory."""
    return tmp_path / "test_results"

@pytest.fixture
def mock_signal():
    """Fixture for mocking Signal class."""
    with patch('src.simulation.environment.Signal') as mock:
        mock_signal = MagicMock(spec=Signal)
        mock_signal.lanes = ["lane1", "lane2"]
        mock_signal.phase = 0  # Current active phase
        mock_signal.full_observation = {
            "lane1": {
                "total_wait": 10,
                "queue": 2,
                "approach": 3,
                "vehicles": [
                    {"id": "veh1", "speed": 10.0},
                    {"id": "veh2", "speed": 15.0}
                ]
            },
            "lane2": {
                "total_wait": 15,
                "queue": 3,
                "approach": 4,
                "vehicles": [
                    {"id": "veh1", "speed": 12.0},
                    {"id": "veh2", "speed": 18.0}
                ]
            }
        }
        mock_signal.observe.return_value = None
        mock_signal.prep_phase.return_value = None
        mock_signal.set_phase.return_value = None
        # Initialize signals dictionary to avoid KeyError
        mock_signal.signals = {}
        mock.return_value = mock_signal
        yield mock_signal

@pytest.fixture
def mock_traci():
    """Fixture for mocking traci module."""
    mock = MagicMock()
    mock.connection = MagicMock()
    mock.connection.has.return_value = False
    mock.connection.get.return_value = MagicMock()
    mock.connection.switch.return_value = None
    mock.start.return_value = None
    mock.close.return_value = None
    mock.getConnection.return_value = mock.connection
    mock.vehicle = MagicMock()
    mock.vehicle.getIDList.return_value = ["veh1", "veh2"]
    mock.vehicle.getCO2Emission.return_value = 0.5
    mock.vehicle.getFuelConsumption.return_value = 0.3
    mock.vehicle.getNoiseEmission.return_value = 0.4
    mock.simulation = MagicMock()
    mock.simulation.getTime.return_value = 0
    return mock

@pytest.fixture
def mock_get_traci(mock_traci):
    """Fixture for mocking get_traci function to return our mocked traci."""
    with patch('src.utils.get_traci', return_value=mock_traci):
        yield mock_traci

@pytest.fixture
def mock_sumo_env(mock_traci, mock_signal):
    """Fixture for mocked SUMO environment."""
    with patch('src.simulation.environment.SumoEnvironment') as mock:
        mock_env = MagicMock(spec=SumoEnvironment)
        mock_env.connection = mock_traci
        mock_env.signal_ids = ["1", "2"]
        mock_env.active_signal_ids = ["1", "2"]
        mock_env.signal_phases = {
            "1": ["phase1", "phase2"],
            "2": ["phase1", "phase2"]
        }
        # Configure get_time to return an integer value
        mock_env.get_time.return_value = 100  # Return a value less than end_time (3600)
        mock_env.get_vehicle_ids.return_value = ["veh1", "veh2"]
        mock_env.get_vehicle_emissions.return_value = 0.5
        mock_env.get_fuel_consumption.return_value = 0.3
        mock_env.get_noise_emission.return_value = 0.4
        mock_env.close.return_value = None
        mock_env.reset.return_value = None
        mock_env.step.return_value = None
        mock_env._identify_traffic_signals.return_value = None
        mock_env._setup_active_signals.return_value = None
        mock.return_value = mock_env
        yield mock_env

@pytest.fixture
def mock_normalize_traffic_state():
    """Fixture for mocking normalize_traffic_state function."""
    with patch('src.components.states.normalize_traffic_state') as mock:
        mock.return_value = {
            "1": np.zeros((10,)),
            "2": np.zeros((10,))
        }
        yield mock

@pytest.fixture
def mock_rl_reward():
    """Fixture for mocking rl_reward function."""
    with patch('src.components.rewards.rl_reward') as mock:
        mock.return_value = {
            "1": 10,
            "2": 15
        }
        yield mock

@pytest.fixture
def mock_sumolib():
    """Fixture for creating a mock sumolib object."""
    mock = MagicMock()
    mock.checkBinary = lambda x: x
    return mock

# Patch the SumoEnvironment.get_time method directly
@pytest.fixture(autouse=True)
def patch_sumo_env_get_time():
    """Fixture to patch the SumoEnvironment.get_time method."""
    with patch('src.config.environment_config.SumoEnvironment.get_time', return_value=100):
        yield
