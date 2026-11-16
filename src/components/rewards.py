import numpy as np
from src.config.simulation_config import RewardParameters

class RewardCalculator:
    """
    Calculates rewards for RL traffic signal control.
    """

    def __init__(self, params: RewardParameters = RewardParameters()):
        self.params = params

    def calculate_reward(
        self,
        avg_waiting_time,
        vehicle_throughput,
        journeys_completed,
        queue_length,
    ):
        # Calculate penalties (negative components)
        waiting_time_penalty_component = -self.params.waiting_time_penalty * avg_waiting_time
        queue_penalty_component = -self.params.queue_penalty * queue_length

        # Calculate rewards (positive components)
        throughput_reward_component = self.params.throughput_reward * vehicle_throughput
        journey_reward_component = self.params.journey_reward * journeys_completed

        combined_reward = (
            waiting_time_penalty_component +
            queue_penalty_component +
            throughput_reward_component +
            journey_reward_component
        )

        # Remove temporary variables
        del waiting_time_penalty_component, queue_penalty_component, throughput_reward_component, journey_reward_component

        # Convert to integer before returning
        return round(combined_reward)


def rl_reward(signals, *, params: RewardParameters = RewardParameters()):
    """
    Calculate rewards for all traffic signals in the network.
    Rewards are normalized by the total number of vehicles in the network to maintain consistency
    """
    reward_calculator = RewardCalculator(params)

    total_vehicle_count = _get_total_vehicle_count(signals)

    signal_rewards = {}

    # Calculate reward for each signal
    for signal_id, signal in signals.items():
        if not signal.lanes:
            signal_rewards[signal_id] = 0
            continue

        metrics = _calculate_signal_metrics(signal)

        raw_reward = reward_calculator.calculate_reward(
            metrics["avg_waiting_time"],
            metrics["vehicle_throughput"],
            metrics["journeys_completed"],
            metrics["total_queue"]
        )

        # Normalize reward
        signal_rewards[signal_id] = round(raw_reward / total_vehicle_count)

    return signal_rewards


def _get_total_vehicle_count(signals):
    """
    Count the total number of vehicles in the network
    """
    for signal in signals.values():
        if "num_vehicles" in signal.full_observation:
            vehicles = signal.full_observation["num_vehicles"]
            if vehicles:  # If vehicles list is not empty
                return len(vehicles)
    # If no vehicles are found, return 1
    return 1


def _calculate_signal_metrics(signal):
    """
    Calculate traffic metrics for a single traffic signal.
    """
    # Extract lane data
    lane_data = [signal.full_observation[lane] for lane in signal.lanes]

    total_waiting_time = np.sum([data["total_wait"] for data in lane_data])
    total_queue = np.sum([data["queue"] for data in lane_data])
    lane_count = len(lane_data)

    # Get vehicle throughput and journey completion metrics
    departures = signal.full_observation.get("departures", [])
    arrivals = signal.full_observation.get("arrivals", [])

    metrics = {
        "avg_waiting_time": total_waiting_time / max(lane_count, 1),
        "vehicle_throughput": len(departures),
        "journeys_completed": len(arrivals),
        "total_queue": total_queue
    }

    # Remove temporary variables
    del lane_data, total_waiting_time, total_queue, lane_count, departures, arrivals

    return metrics

def calculate_wait_times(signals):
    """
    Calculate average waiting times for all signals in the network
    """
    wait_time_dict = {}

    # Calculate average wait time for each signal
    for signal_id, signal in signals.items():
        if not hasattr(signal, 'lane_manager') or not signal.lane_manager.lanes:
            wait_time_dict[signal_id] = 0.0
            continue

        # Sum waiting times across all lanes controlled by this signal
        total_wait = sum(signal.full_observation[lane]["total_wait"] for lane in signal.lane_manager.lanes)

        # Calculate average
        lane_count = len(signal.lane_manager.lanes)
        wait_time_dict[signal_id] = total_wait / max(lane_count, 1)

        # Remove temporary variables
        del total_wait, lane_count

    return wait_time_dict
