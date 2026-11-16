"""Train, evaluate and export the NTN dashboard results."""

from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory

from src.smoke import build_synthetic_scenario

from src.config.simulation_config import SimulationConfig
from src.visualization.plotting import plot_results
from src.utils.processing import (
    export_metrics_to_json,
    process_testing_metrics,
    run_episode,
    run_training_episodes,
)
from src.utils.simulation_utils import (
    initialize_environment,
    initialize_agent,
    get_connection_name,
    get_model_dir,
)


def main(simulation_config: SimulationConfig):
    if simulation_config.map_name == "synthetic-cross":
        with TemporaryDirectory(prefix="ntn-dashboard-") as directory:
            workspace = Path(directory)
            build_synthetic_scenario(workspace)
            config = replace(
                simulation_config, project_root=workspace,
                map_config=dict(simulation_config.map_config, net="cross.sumocfg"),
            )
            return run_dashboard(config)
    if not (simulation_config.project_root / simulation_config.map_config["net"]).is_file():
        raise FileNotFoundError("Install the scenario with python scripts/install_ingolstadt21.py")
    return run_dashboard(simulation_config)


def run_dashboard(simulation_config: SimulationConfig):
    """
    training, testing of both the RL and baseline agents, and visualization of performance metrics.
    """
    simulation_config.results_directory.mkdir(parents=True, exist_ok=True)
    training_metrics = {}
    # Training phase
    if not simulation_config.load_model and not simulation_config.load_pretrained_model:
        print("Training RL Model")
        training_metrics = train_model(simulation_config, simulation_config.results_directory)

    # Testing phase
    print(f"Testing Phase: {"Pretrained Model" if simulation_config.load_pretrained_model else "Trained Model"}")
    testing_results = testing(simulation_config, simulation_config.pretrained_models_directory if simulation_config.load_pretrained_model else get_model_dir(simulation_config.results_directory) / simulation_config.map_name)

    print("Plotting Results")
    plot_results(
        simulation_config=simulation_config,
        training_metrics=training_metrics if (not simulation_config.load_model and not simulation_config.load_pretrained_model) else {},
        testing_results=testing_results,
        results_dir=simulation_config.results_directory
    )

    print("Exporting metrics to JSON")
    export_metrics_to_json(simulation_config, training_metrics, testing_results)


def train_model(simulation_config: SimulationConfig, training_dir: Path) -> dict:
    """
    Train the reinforcement learning model using only the first 30 minutes of simulation time.
    """
    start = simulation_config.map_config["start_time"]
    midpoint = (start + simulation_config.map_config["end_time"]) // 2
    training_env = initialize_environment(
        simulation_config=simulation_config,
        connection_name="rl_train",
        start_time=start,
        end_time=midpoint,
    )

    # Initialize the multi-agent RL controller
    training_agent = initialize_agent(
        simulation_config=simulation_config,
        environment=training_env,
        is_rl_agent=True
    )

    try:
        training_metrics = run_training_episodes(
            env=training_env, agent=training_agent,
            num_episodes=simulation_config.training_episodes,
        )
    finally:
        training_env.close()

    model_path = get_model_dir(training_dir) / simulation_config.map_name
    training_agent.save(model_path)
    print(f"Training complete. Model saved to {model_path}")

    return training_metrics


def testing(simulation_config: SimulationConfig, model_dir: Path) -> dict:
    """ Test RL and non-RL agents under different traffic conditions """

    testing_results = {}

    # Define time periods
    end_time = simulation_config.map_config["end_time"]
    step = simulation_config.map_config["step_length"]
    time_period = (end_time - simulation_config.map_config["start_time"]) // (6 * step) * step
    start_time = end_time - 3 * time_period

    testing_periods = [
        {"start_time": start_time, "end_time": start_time + time_period,  "name": "light",  "scale": 0.5},
        {"start_time": start_time + time_period, "end_time": start_time + 2 * time_period, "name": "normal", "scale": 1.0},
        {"start_time": start_time + 2 * time_period, "end_time": start_time + 3 * time_period, "name": "heavy", "scale": 3.0}
    ]

    # Clean up temporary variables
    del start_time, time_period

    # Test both RL and non-RL controllers
    for using_rl_agent in [True, False]:
        agent_type = get_connection_name(using_rl_agent)
        testing_results[agent_type] = {"metrics": {}}

        print(f"\nTesting {agent_type} model")

        for time_period in testing_periods:
            print(f"\nTesting with traffic demand with scale {time_period['scale']}")

            test_env = initialize_environment(
                simulation_config=simulation_config,
                connection_name=f"{agent_type}_test_{time_period['name']}",
                start_time=time_period['start_time'],
                end_time=time_period['end_time'],
                scale=time_period['scale']
            )

            test_agent = initialize_agent(
                simulation_config=simulation_config,
                environment=test_env,
                is_rl_agent=using_rl_agent,
                model_dir=model_dir if using_rl_agent else None
            )

            try:
                testing_episode_metrics = run_episode(test_env, test_agent)
            finally:
                test_env.close()

            testing_metrics = process_testing_metrics(testing_episode_metrics)
            testing_results[agent_type]["metrics"][time_period['name']] = testing_metrics

            episode_reward = sum(step_metric.rewards for step_metric in testing_episode_metrics)
            print(f"Testing completed for {time_period['name']} traffic with reward: {episode_reward:.2f}")

            # Clean up temporary variables
            del test_env, test_agent, testing_episode_metrics, testing_metrics, episode_reward

    return testing_results
