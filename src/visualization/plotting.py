import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Union, Callable
import shutil
from src.utils.processing import (
    get_episode_metrics,
    prepare_dataframe,
    calculate_optimal_params,
    convert_metrics_to_list,
)
matplotlib.use("agg")

# Configure consistent visualization style for all plots in this module
plt.style.use("ggplot")
sns.set_theme(style="darkgrid")

PALETTE = sns.color_palette("muted", desat=0.9)

# Set global matplotlib parameters
plt.rcParams['legend.title_fontsize'] = 16
plt.rcParams['figure.titlesize'] = 20
plt.rcParams['figure.titleweight'] = 'bold'
plt.rcParams['axes.titleweight'] = 'bold'
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['axes.titlesize'] = 18
plt.rcParams['axes.labelsize'] = 16
plt.rcParams['xtick.labelsize'] = 14
plt.rcParams['ytick.labelsize'] = 14
plt.rcParams['legend.fontsize'] = 14

def apply_smoothing(y, smooth_factor):
    """
    Apply moving average smoothing to a time series
    """
    # Calculate smoothed values with moving average
    y_smooth = np.convolve(y, np.ones(smooth_factor) / smooth_factor, mode="valid")

    pad = len(y) - len(y_smooth)
    return np.pad(y_smooth, (pad, 0), "edge")


def handle_plot_saving_and_display(fig, save_path=None, dpi=100):
    # Save figure if a path is provided
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)  # Close the figure to free memory


def plot_metric(
    metrics,
    metric_name,
    x_label="",
    x_range=0,
    title="",
    save_path=None,
    smooth_factor=0,
    figsize=(10, 6),
    dpi=100,
    auto_smooth=True,
    color_index=0,
    y_label="",
):
    fig, ax = plt.subplots(figsize=figsize)
    x, y = get_episode_metrics(metrics, metric_name, x_range=x_range)

    if auto_smooth and smooth_factor == 0 and len(x) > 100:
        params = calculate_optimal_params(len(x))
        smooth_factor = params["smooth_factor"]
    else:
        params = calculate_optimal_params(len(x))

    # Apply smoothing
    if smooth_factor > 0:
        y_smooth = apply_smoothing(y, smooth_factor)

        if len(x) < 5000:
            ax.plot(x, y, alpha=0.3, color=PALETTE[color_index], label="Raw Data", linewidth=1.5)

        ax.plot(x, y_smooth, linewidth=2.5, color=PALETTE[color_index], label="Smoothed Data")

    else:
        label = metric_name.replace("_", " ").title()
        ax.plot(x, y, color=PALETTE[color_index], linewidth=2.5, label=label)


    ax.set_xlabel(x_label if x_label else "Episode")
    ax.set_ylabel(y_label if y_label else label)

    # Add dataset size info to title for large datasets
    if len(x) > 1000:
        title_suffix = f" ({len(x)} Episodes)"
        if title and title[-1] == ")":
            title = title[:-1] + title_suffix[1:]
        else:
            title = title + title_suffix

    ax.set_title(title, pad=15)

    ax.legend(frameon=True, framealpha=0.9, loc='best')
    ax.grid(True, linestyle="--", alpha=0.7)

    plt.tight_layout()
    handle_plot_saving_and_display(fig, save_path, dpi)

    return fig, ax


# Define metric configurations for all supported metrics
METRIC_CONFIGS = {
    "rewards": {
        "title": "Model Improvement Over Time",
        "y_label": "Aggregated Reward",
        "color_index": 0
    },
    "waiting_times": {
        "title": "Average Wait Time Per Episode",
        "y_label": "Average Wait Time (s)",
        "color_index": 1
    },
    "queue_lengths": {
        "title": "Average Queue Lengths Per Episode",
        "y_label": "Average Queue Length",
        "color_index": 2
    },
    "co2_emissions": {
        "title": "CO2 Emissions Per Episode",
        "y_label": "Average CO2 Emissions (g)",
        "color_index": 3
    },
    "avg_speeds": {
        "title": "Average Speeds Per Episode",
        "y_label": "Average Speed (km/h)",
        "color_index": 4
    },
    "fuel_consumption": {
        "title": "Fuel Consumption Per Episode",
        "y_label": "Average Fuel Consumption (L/min)",
        "color_index": 5
    },
    "noise_emission": {
        "title": "Noise Emission Per Episode",
        "y_label": "Noise Emission (dB)",
        "color_index": 6
    }
}

def create_metric_plot(
    metrics,
    metric_name,
    x_label="",
    x_range=0,
    title=None,
    save_path=None,
    smooth_factor=0,
    figsize=(10, 6),
    dpi=100,
    auto_smooth=True,
    y_label=None,
):
    if metric_name not in METRIC_CONFIGS:
        raise ValueError(f"Unsupported metric: {metric_name}")

    config = METRIC_CONFIGS[metric_name]

    title = title if title is not None else config["title"]
    y_label = y_label if y_label is not None else config["y_label"]

    result = plot_metric(
        metrics=metrics,
        metric_name=metric_name,
        x_label=x_label,
        x_range=x_range,
        title=title,
        save_path=save_path,
        smooth_factor=smooth_factor,
        figsize=figsize,
        dpi=dpi,
        auto_smooth=auto_smooth,
        color_index=config["color_index"],
        y_label=y_label,
    )

    del config, title, y_label
    return result

# Helper function to generate metric-specific plotting functions dynamically
def generate_metric_function(metric_name: str) -> Callable:
    def plot_function(
        metrics, x_label="", x_range=0, title=None, save_path=None,
        smooth_factor=0, figsize=(10, 6), dpi=100, auto_smooth=True
    ):
        return create_metric_plot(
            metrics, metric_name, x_label, x_range, title, save_path,
            smooth_factor, figsize, dpi, auto_smooth
        )
    return plot_function

# Generate plot functions for all metrics defined in METRIC_CONFIGS
plot_rewards = generate_metric_function("rewards")
plot_wait_time = generate_metric_function("waiting_times")
plot_queue_lengths = generate_metric_function("queue_lengths")
plot_co2_emissions = generate_metric_function("co2_emissions")
plot_avg_speeds = generate_metric_function("avg_speeds")
plot_fuel_consumption = generate_metric_function("fuel_consumption")
plot_noise_emission = generate_metric_function("noise_emission")


def plot_metric_comparison(
    metrics_list: List[List[Dict]],
    metric_name: str,
    labels: Optional[List[str]] = None,
    x_label: str = "Episode",
    y_label: Optional[str] = None,
    x_range: int = 0,
    title: Optional[str] = None,
    smooth_factor: int = 0,
    save_path: Optional[Union[str, Path]] = None,
    figsize: Tuple[int, int] = (10, 6),
    dpi: int = 100,
    auto_smooth: bool = True,
):
    """
    compare different models side-by-side
    """

    if title is None or y_label is None:
        # Use METRIC_CONFIGS for consistent naming if available
        if metric_name in METRIC_CONFIGS:
            config = METRIC_CONFIGS[metric_name]
            default_title = f"{config['title'].replace(' Per Episode', '')} Comparison"
            default_y_label = config['y_label']
        else:
            # Fallback if metric not in METRIC_CONFIGS
            default_title = f"{metric_name.replace('_', ' ').title()} Comparison"
            default_y_label = metric_name.replace('_', ' ').title()

        title = title or default_title
        y_label = y_label or default_y_label

    # Handle case of single metrics dict
    if not isinstance(metrics_list[0], list): metrics_list = [metrics_list]

    fig, ax = plt.subplots(figsize=figsize)

    # Calculate total data size and determine downsample factor
    total_points = sum(
        min(x_range if x_range > 0 else len(metrics), len(metrics))
        for metrics in metrics_list
    )

    df = prepare_dataframe(metrics_list, metric_name, labels, x_range)

    # Apply smoothing
    if auto_smooth and smooth_factor == 0:
        max_episodes = max(df["Episode"])
        if max_episodes > 100:
            params = calculate_optimal_params(max_episodes)
            smooth_factor = params["smooth_factor"]

    # Apply smoothing
    if smooth_factor > 0:
        # Group by run and apply rolling mean for smoothing
        df_grouped = df.groupby(["Run", "Episode"])[metric_name].mean().reset_index()
        smooth_column = f"{metric_name}_smooth"
        df_grouped[smooth_column] = df_grouped.groupby("Run")[metric_name].transform(
            lambda x: x.rolling(smooth_factor, min_periods=1).mean()
        )
        metric_to_plot = smooth_column
    else:
        df_grouped = df
        metric_to_plot = metric_name

    # Count unique runs
    unique_runs = df_grouped["Run"].nunique()
    limited_palette = PALETTE[:unique_runs]

    # Create line plots
    for i, (run_name, run_data) in enumerate(df_grouped.groupby("Run")):
        color = limited_palette[i]
        # Plot
        ax.plot(run_data["Episode"], run_data[metric_to_plot],
                color=color, linewidth=2.5, label=run_name)

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title, pad=15)

    # Add subtle grid
    ax.grid(True, linestyle="--", alpha=0.7)

    ax.legend(
        title="Agent",
        loc="best",
        frameon=True,
        framealpha=0.9
    )

    plt.tight_layout()
    handle_plot_saving_and_display(fig, save_path, dpi)

    return fig, ax


def plot_metrics_grid(
    metrics: List[Dict],
    metrics_to_plot=None,
    x_range: int = 0,
    title: str = "Performance Metrics",
    smooth_factor: int = 0,
    save_path: Optional[Union[str, Path]] = None,
    figsize: Tuple[int, int] = (15, 10),
    dpi: int = 100,
    auto_smooth: bool = True,
):
    """
    Create a grid of plots showing multiple performance metrics together.
    """
    if metrics_to_plot is None:
        metrics_to_plot = [
            ("rewards", "Rewards", "Aggregated Reward"),
            ("waiting_times", "Wait Times", "Average Wait Time (s)"),
            ("queue_lengths", "Queue Lengths", "Average Queue Length"),
        ]

    # Create figure with N vertically stacked subplots sharing x-axis
    num_metrics = len(metrics_to_plot)
    fig, axs = plt.subplots(num_metrics, 1, figsize=figsize, sharex=True)

    if num_metrics == 1: axs = [axs]

    if x_range == 0: x_range = len(metrics)

    if auto_smooth and smooth_factor == 0 and x_range > 100:
        params = calculate_optimal_params(x_range)
        smooth_factor = params["smooth_factor"]

    # Plot each metric in its own subplot
    for i, (metric_name, subplot_title, y_label) in enumerate(metrics_to_plot):
        x, y = get_episode_metrics(metrics, metric_name, x_range)

        params = calculate_optimal_params(len(x))
        color_idx = i % len(PALETTE)

        if smooth_factor > 0:
            y_smooth = apply_smoothing(y, smooth_factor)
            if len(x) < 5000:
                axs[i].plot(x, y, alpha=0.3, color=PALETTE[color_idx], label="Raw", linewidth=1.5)

            axs[i].plot(x, y_smooth, linewidth=2.5, color=PALETTE[color_idx], label="Smoothed")
        else:
            axs[i].plot(x, y, linewidth=2.5, color=PALETTE[color_idx])

        axs[i].set_title(subplot_title, pad=10)
        axs[i].set_ylabel(y_label)
        axs[i].grid(True, linestyle="--", alpha=0.7)

        if smooth_factor > 0 and len(x) < 5000:
            axs[i].legend(loc="best", frameon=True, framealpha=0.9)

    axs[-1].set_xlabel("Episode")

    if x_range > 1000:
        title_suffix = f" ({x_range} Episodes)"
        title = (title[:-1] + title_suffix[1:] if title[-1] == ")" else title + title_suffix)

    fig.suptitle(title, y=0.98)

    plt.tight_layout()
    fig.subplots_adjust(top=0.93)

    handle_plot_saving_and_display(fig, save_path, dpi)

    return fig


def create_plot_directories(results_dir: Path) -> Dict[str, Path]:
    """
    Create the directory structure for plots and metrics.
    """

    directories = {
        'metrics': results_dir / "metrics",
        'plots': results_dir / "plots",
        'rl_model_weights': results_dir / "rl_model_weights",
        'rl_train': results_dir / "plots" / "rl_train",
        'comparisons': results_dir / "plots" / "comparisons"
    }

    # Create all directories
    for directory in directories.values():
        directory.mkdir(parents=True, exist_ok=True)

    return directories

def plot_results(
    simulation_config,
    training_metrics: Dict,
    testing_results: Dict,
    results_dir: Path
) -> None:
    """
    Create plots of training and testing metrics.
    """
    # Create directory structure
    directories = create_plot_directories(results_dir)

    # Only clear and recreate directories that are relevant to the current operation
    if not simulation_config.load_model and training_metrics:
        # Clear and recreate training directory only if we're training
        if directories['rl_train'].exists():
            shutil.rmtree(directories['rl_train'])
        directories['rl_train'].mkdir(parents=True, exist_ok=True)

        create_training_plots(
            training_metrics=training_metrics,
            plots_dir=directories['rl_train'],
            num_episodes=simulation_config.training_episodes
        )

    # Clear and recreate comparisons directory
    if directories['comparisons'].exists():
        shutil.rmtree(directories['comparisons'])
    directories['comparisons'].mkdir(parents=True, exist_ok=True)

    # Create comparison plots
    create_comparison_plots(testing_results, directories['comparisons'])


def create_training_plots(training_metrics: Dict, plots_dir: Path, num_episodes: int):
    """
    Create plots showing training progress over episodes.
    """
    # Training performance plots
    performance_plots = [
        (plot_rewards, "rewards", "Rewards"),
        (plot_wait_time, "waiting_times", "Average Wait Time"),
        (plot_queue_lengths, "queue_lengths", "Average Queue Length"),
        (plot_co2_emissions, "co2_emissions", "CO2 Emissions"),
        (plot_fuel_consumption, "fuel_consumption", "Fuel Consumption"),
        (plot_noise_emission, "noise_emission", "Noise Emission"),
        (plot_avg_speeds, "avg_speeds", "Average Speeds"),
    ]

    # Create individual metric plots
    for plot_func, metric_name, metric_label in performance_plots:
        plot_func(
            training_metrics,
            x_label="Episode",
            x_range=num_episodes,
            title=f"RL Model Training Phase {metric_label}",
            save_path=plots_dir / f"training_{metric_name}.png",
        )

def create_comparison_plots(testing_results: Dict, comparison_dir: Path):
    """
    Create comparison plots between RL and baseline agents for different traffic conditions.
    """
    # Traffic conditions to compare
    conditions_dict = {"light": "Light Traffic", "normal": "Normal Traffic", "heavy": "Heavy Traffic"}

    for condition in conditions_dict.keys():
        rl_metrics = convert_metrics_to_list(testing_results["rl"]["metrics"][condition])
        non_rl_metrics = convert_metrics_to_list(testing_results["non_rl"]["metrics"][condition])

        # List of metrics to compare
        metrics_to_compare = [
            (
                "rewards",
                f"Reward Comparison: RL vs. Baseline ({conditions_dict[condition]})",
                "Reward Value",
            ),
            (
                "waiting_times",
                f"Waiting Time Comparison: RL vs. Baseline ({conditions_dict[condition]})",
                "Average Wait Time (s)",
            ),
            (
                "queue_lengths",
                f"Queue Length Comparison: RL vs. Baseline ({conditions_dict[condition]})",
                "Average Queue Length",
            ),
            (
                "co2_emissions",
                f"CO2 Emissions Comparison: RL vs. Baseline ({conditions_dict[condition]})",
                "Average CO2 Emissions (g)",
            ),
            (
                "avg_speeds",
                f"Average Speed Comparison: RL vs. Baseline ({conditions_dict[condition]})",
                "Average Speed (km/h)",
            ),
            (
                "fuel_consumption",
                f"Fuel Consumption Comparison: RL vs. Baseline ({conditions_dict[condition]})",
                "Average Fuel Consumption (L/min)",
            ),
            (
                "noise_emission",
                f"Noise Emission Comparison: RL vs. Baseline ({conditions_dict[condition]})",
                "Noise Level (dB)",
            ),
        ]

        # Create comparison plots for each metric
        for metric_name, title, y_label in metrics_to_compare:
            plot_metric_comparison(
                metrics_list=[rl_metrics, non_rl_metrics],
                metric_name=metric_name,
                labels=["RL Agent", "Baseline Agent"],
                x_label="Simulation Step",
                y_label=y_label,
                title=title,
                save_path=comparison_dir / f"{metric_name}_{condition}_comparison.png",
                auto_smooth=True,
            )
