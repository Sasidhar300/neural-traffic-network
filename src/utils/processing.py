from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
from src.utils.simulation_utils import (
    calculate_average,
    save_json
)


def run_episode(env, agent) -> List:
    """
    Run an episode in the traffic simulation environment.
    """
    observations = env.reset()
    episode_metrics = []

    while not env.is_simulation_complete:
        actions = agent.act(observations)
        observations,rewards = env.step(actions)
        step_metrics = env.get_metrics()
        episode_metrics.append(step_metrics)
        if agent.training:
            agent.observe(observations, rewards, env.is_simulation_complete)

    return episode_metrics


def export_metrics_to_json(
    simulation_config,
    training_metrics: Dict,
    testing_results: Dict
) -> None:
    """
    Export all metrics from training and testing to a JSON file.
    """
    # Create directory if it doesn't exist
    metrics_dir = simulation_config.results_directory / "metrics"
    metrics_dir.mkdir(exist_ok=True)

    def aggregate_metrics(metrics_dict):
        metric_keys = [
            "rewards", "waiting_times", "queue_lengths", "co2_emissions",
            "avg_speeds", "fuel_consumption", "noise_emission"
        ]

        aggregated = {}
        for metric in metric_keys:
            values = [entry[metric] for entry in metrics_dict.values()]
            aggregated[metric] = np.mean(values)

        return aggregated

    # Aggregate metrics for training and testing
    metrics_data = {}
    if training_metrics:
        metrics_data["training"] = aggregate_metrics(training_metrics)
    metrics_data["testing"] = {
        "rl": {
            "light": aggregate_metrics(testing_results["rl"]["metrics"]["light"]),
            "normal": aggregate_metrics(testing_results["rl"]["metrics"]["normal"]),
            "heavy": aggregate_metrics(testing_results["rl"]["metrics"]["heavy"])
        },
        "baseline": {
            "light": aggregate_metrics(testing_results["non_rl"]["metrics"]["light"]),
            "normal": aggregate_metrics(testing_results["non_rl"]["metrics"]["normal"]),
            "heavy": aggregate_metrics(testing_results["non_rl"]["metrics"]["heavy"])
        }
    }

    # Use utility function to save JSON
    metrics_file = metrics_dir / "performance_metrics.json"
    save_json(metrics_data, metrics_file)

    print(f"Metrics exported to {metrics_file}")


def aggregate_episode_metrics(episode_metrics: List) -> Dict:
    """
    Aggregate metrics
    """
    metrics_to_aggregate = [
        "waiting_times", "queue_lengths", "co2_emissions",
        "avg_speeds", "fuel_consumption", "noise_emission"
    ]

    total_rewards = 0
    metric_sums = {metric: 0 for metric in metrics_to_aggregate}
    metric_counts = {metric: 0 for metric in metrics_to_aggregate}

    # Process metrics from each simulation step
    for step_metrics in episode_metrics:
        total_rewards += step_metrics.rewards

        for metric in metrics_to_aggregate:
            metric_value = getattr(step_metrics, metric, None)

            if metric_value:
                # Calculate average for this metric and add to sum
                metric_sums[metric] += sum(metric_value.values()) / len(metric_value)
                metric_counts[metric] += 1

    # Calculate averages
    result = {"rewards": total_rewards}
    for metric in metrics_to_aggregate:
        result[metric] = metric_sums[metric] / max(1, metric_counts[metric])

    # Remove temporary variables
    del metrics_to_aggregate, metric_sums, metric_counts

    return result


def process_testing_metrics(testing_episode_metrics: List) -> Dict:
    """
    Process testing metrics for visualization.
    """
    testing_metrics = {}

    for step, step_metric in enumerate(testing_episode_metrics):
        testing_metrics[step] = {
            "rewards": step_metric.rewards,
            "waiting_times": calculate_average(step_metric.waiting_times),
            "queue_lengths": calculate_average(step_metric.queue_lengths),
            "co2_emissions": calculate_average(step_metric.co2_emissions),
            "avg_speeds": calculate_average(step_metric.avg_speeds),
            "fuel_consumption": calculate_average(step_metric.fuel_consumption),
            "noise_emission": calculate_average(step_metric.noise_emission),
        }

    del step, step_metric
    return testing_metrics


def run_training_episodes(env, agent, num_episodes: int) -> Dict:
    """
    Run multiple training episodes and collect metrics.
    """
    training_metrics = {}

    # Run for the number of episodes specified in the config
    for episode in range(num_episodes):
        episode_metrics = run_episode(env, agent)

        aggregated_metrics = aggregate_episode_metrics(episode_metrics)
        episode_metrics.clear()
        training_metrics[episode] = aggregated_metrics

        print(f"Finished episode {episode + 1} with reward: {aggregated_metrics['rewards']}")

    return training_metrics


def get_episode_metrics(metrics: Dict, metric_name: str, x_range: int = 0) -> Tuple[List[int], List[float]]:
    """
    Extract specific metric data for plotting.
    """
    if x_range == 0:
        x_range = len(metrics)

    return (
        list(range(x_range)),
        [metrics[episode_num].get(metric_name, 0) for episode_num in range(x_range)],
    )


def prepare_dataframe(metrics_list: List[Dict], metric_name: str, labels: Optional[List[str]] = None, x_range: int = 0) -> pd.DataFrame:
    """
    Convert metrics into a pandas DataFrame.
    """
    df_data = []

    for i, metrics in enumerate(metrics_list):
        x, y = get_episode_metrics(metrics, metric_name, x_range)

        run_label = labels[i] if labels and i < len(labels) else f"Run {i+1}"

        for step, value in zip(x, y):
            df_data.append({"Episode": step, metric_name: value, "Run": run_label})

    return pd.DataFrame(df_data)


def calculate_optimal_params(x_range: int) -> Dict:
    """
    Calculate optimal smoothing parameters based on dataset size.
    """
    if x_range < 50:
        smooth_factor = 3
    elif x_range < 200:
        smooth_factor = 5
    elif x_range < 1000:
        smooth_factor = 15
    else:
        smooth_factor = 31

    return {"smooth_factor": smooth_factor}


def convert_metrics_to_list(metrics_dict: Dict) -> List[Dict]:
    """
    Convert metrics dictionary to a list format for processing.
    """
    return [metrics_dict[step] for step in sorted(metrics_dict.keys())]
