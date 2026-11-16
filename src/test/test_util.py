import numpy as np
import torch
import random
from pathlib import Path
from unittest.mock import Mock, patch
import sys
import json

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from src.utils.simulation_utils import (
    set_random_seeds,
    get_connection_name,
    get_model_dir,
    prepare_directory,
    calculate_average_metric,
    calculate_emission_metric,
    start_sumo_connection,
    close_sumo_connection,
)
from src.utils.processing import save_json, calculate_average

# Import fixtures from test/utils.py
from src.test.utils import (
    simulation_config,
    test_dir,
    mock_sumo_env,
    mock_traci,
    mock_sumolib
)

def test_set_random_seeds():
    """Test that random seeds are properly set for reproducibility."""
    seed = 42

    # Get initial random numbers
    np.random.seed(seed)
    initial_np = np.random.randint(1000)
    torch.manual_seed(seed)
    initial_torch = torch.randint(1000, (1,)).item()
    random.seed(seed)
    initial_py = random.randint(0, 1000)

    # Call set_random_seeds
    set_random_seeds(seed)

    # Get new random numbers
    new_np = np.random.randint(1000)
    new_torch = torch.randint(1000, (1,)).item()
    new_py = random.randint(0, 1000)

    # Verify that set_random_seeds properly set the seeds
    assert new_np == initial_np, "NumPy random seed not properly set"
    assert new_torch == initial_torch, "PyTorch random seed not properly set"
    assert new_py == initial_py, "Python random seed not properly set"

def test_get_connection_name():
    """Test that connection name is correctly returned based on model usage."""
    assert get_connection_name(True) == "rl"
    assert get_connection_name(False) == "non_rl"

def test_prepare_directory(test_dir):
    """Test directory preparation with and without cleaning."""
    test_path = test_dir / "test_prepare"

    # Test creating new directory
    prepare_directory(test_path, clean=False)
    assert test_path.exists()

    # Create a test file
    test_file = test_path / "test.txt"
    test_file.write_text("test")

    # Test cleaning directory
    prepare_directory(test_path, clean=True)
    assert test_path.exists()
    assert not test_file.exists()

def test_get_model_dir():
    """Test that model directory path is correctly constructed."""
    training_dir = Path("test_training")
    expected_path = training_dir / "rl_model_weights"

    result = get_model_dir(training_dir)
    assert result == expected_path

def test_save_json(tmp_path):
    """Test that save_json correctly saves data to a JSON file."""
    # Test data
    data = {
        "test_key": "test_value",
        "nested": {
            "key1": 1,
            "key2": 2
        },
        "list": [1, 2, 3]
    }

    # Create a file path
    file_path = tmp_path / "test_dir" / "test.json"
    result = save_json(data, file_path)

    assert result is True
    assert file_path.exists()
    assert file_path.parent.exists()

    # Verify file contents
    with open(file_path, 'r') as f:
        loaded_data = json.load(f)

    assert loaded_data == data

def test_calculate_average():
    """Test the calculate_average function with various inputs."""

    assert calculate_average({}) == 0.0
    assert calculate_average({"key1": 10.0}) == 10.0

    data = {"key1": 10.0, "key2": 20.0, "key3": 30.0}
    expected = (10.0 + 20.0 + 30.0) / 3
    assert calculate_average(data) == expected

    data = {"key1": 1.5, "key2": 2.5, "key3": 3.5}
    expected = (1.5 + 2.5 + 3.5) / 3
    assert calculate_average(data) == expected

    data = {"key1": 0.0, "key2": 0.0}
    assert calculate_average(data) == 0.0

    data = {"key1": -10.0, "key2": 10.0}
    assert calculate_average(data) == 0.0


@patch('src.utils.simulation_utils.sumolib', new_callable=lambda: Mock())
@patch('src.utils.simulation_utils.get_traci')
def test_start_sumo_connection_gui(mock_get_traci, mock_sumolib_module, mock_traci):
    """Test that start_sumo_connection correctly sets up a SUMO connection."""

    mock_get_traci.return_value = mock_traci
    mock_sumolib_module.checkBinary.side_effect = lambda x: x

    # Test parameters
    config_path = Path("test_config.sumocfg")
    connection_name = "test_connection"
    seed = 42
    start_time = 0
    end_time = 3600
    scale = 1.5

    # Test with GUI
    connection = start_sumo_connection(
        config_path=config_path,
        connection_name=connection_name,
        seed=seed,
        start_time=start_time,
        end_time=end_time,
        scale=scale,
        use_gui=True,
        libsumo=False
    )

    mock_traci.start.assert_called_once()
    call_args = mock_traci.start.call_args[0]

    assert "sumo-gui" in call_args[0]
    assert str(config_path) in call_args[0]
    # The function returns the connection object, which is mock_traci.connection
    assert connection == mock_traci.connection

    # Verify that getConnection was called with the connection name
    mock_traci.getConnection.assert_called_once_with(connection_name)

    # Reset the mocks
    mock_traci.reset_mock()
    mock_sumolib_module.reset_mock()


def test_calculate_average_metric():
    """Test the calculate_average_metric function."""
    # Test with empty dictionary
    assert calculate_average_metric({}) == 0.0

    # Test with values and no metric function
    data = {"key1": 10.0, "key2": 20.0, "key3": 30.0}
    expected = (10.0 + 20.0 + 30.0) / 3
    assert calculate_average_metric(data) == expected

    # Test with values and a metric function
    data = {"key1": {"value": 10}, "key2": {"value": 20}}
    metric_fn = lambda x: x["value"]
    expected = (10 + 20) / 2
    assert calculate_average_metric(data, metric_fn) == expected

    # Test with zeros
    data = {"key1": 0.0, "key2": 0.0}
    assert calculate_average_metric(data) == 0.0

    # Test with more complex metric function
    data = {"key1": [1, 2, 3], "key2": [4, 5, 6]}
    metric_fn = lambda x: sum(x) / len(x)
    expected = ((1 + 2 + 3) / 3 + (4 + 5 + 6) / 3) / 2
    assert calculate_average_metric(data, metric_fn) == expected

def test_calculate_emission_metric(mock_traci):
    """Test the calculate_emission_metric function."""
    # Configure mock
    mock_traci.vehicle.getIDList.return_value = ["veh1", "veh2"]
    mock_traci.vehicle.getCO2Emission.side_effect = lambda vid: 500.0
    mock_traci.vehicle.getFuelConsumption.side_effect = lambda vid: 0.5
    mock_traci.vehicle.getNoiseEmission.side_effect = lambda vid: 60.0

    # Test CO2 emissions
    co2 = calculate_emission_metric(mock_traci, ["veh1", "veh2"], "co2")
    assert co2 == 500.0 * 0.001  # 500 mg * scaling factor 0.001 for g

    # Test fuel consumption
    fuel = calculate_emission_metric(mock_traci, ["veh1", "veh2"], "fuel")
    assert fuel == 0.5 * 0.025 * 60 / 1000  # 0.5 * scaling factor for L/min

    # Test noise emissions
    noise = calculate_emission_metric(mock_traci, ["veh1", "veh2"], "noise")
    assert noise == 60.0  # No scaling factor for noise

    # Test with empty vehicle list
    assert calculate_emission_metric(mock_traci, [], "co2") == 0.0

    # Test with unknown emission type (default scaling factor is 1)
    assert calculate_emission_metric(mock_traci, ["veh1", "veh2"], "unknown") == 0.0

@patch('src.utils.simulation_utils.get_traci')
def test_close_sumo_connection(mock_get_traci, mock_traci):
    """Test that close_sumo_connection properly closes the SUMO connection."""
    mock_get_traci.return_value = mock_traci

    # Test closing the connection
    close_sumo_connection()

    # Verify that traci.close was called
    mock_traci.close.assert_called_once()
