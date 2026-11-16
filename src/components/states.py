import numpy as np


def calculate_statistics(values):
    """ Calculate mean and standard deviation """

    array = np.array(values)
    mean = array.mean() if array.size > 0 else 0
    std = max(array.std(), 1) if array.size > 0 else 1

    # Remove temporary variables
    del array

    return mean, std


def normalize_traffic_state(signals):

    """ Perform normalization """

    # List of metrics
    approaching_vehicle_counts = []
    waiting_time_values = []
    queue_length_values = []
    vehicle_speed_sums = []

    # Collect metrics for all lanes at every signal
    for signal in signals.values():
        for lane in signal.lanes:
            lane_data = signal.full_observation[lane]

            approaching_vehicle_counts.append(lane_data["approach"])
            waiting_time_values.append(lane_data["total_wait"])
            queue_length_values.append(lane_data["queue"])

            # Calculate sum of speeds for all vehicles in the lane
            lane_total_speed = sum(
                vehicle["speed"] for vehicle in lane_data["vehicles"]
            )
            vehicle_speed_sums.append(lane_total_speed)
            del lane_total_speed

    # Calculate stats for each metric
    approach_mean, approach_std = calculate_statistics(approaching_vehicle_counts)
    wait_time_mean, wait_time_std = calculate_statistics(waiting_time_values)
    queue_mean, queue_std = calculate_statistics(queue_length_values)
    speed_mean, speed_std = calculate_statistics(vehicle_speed_sums)

    # Remove temporary variables
    del approaching_vehicle_counts, waiting_time_values, queue_length_values, vehicle_speed_sums

    # Create normalized states for each signal
    normalized_states = {}
    for signal_id, signal in signals.items():

        signal_state = []
        active_phase_index = signal.phase

        # Normalize for each lane
        for lane_index, lane in enumerate(signal.lanes):

            lane_features = []
            lane_features.append(1 if lane_index == active_phase_index else 0)

            # Get metrics for this lane
            lane_data = signal.full_observation[lane]

            # Normalize metrics
            normalized_approach = (lane_data["approach"] - approach_mean) / approach_std
            normalized_wait_time = (lane_data["total_wait"] - wait_time_mean) / wait_time_std
            normalized_queue = (lane_data["queue"] - queue_mean) / queue_std

            lane_features.append(normalized_approach)
            lane_features.append(normalized_wait_time)
            lane_features.append(normalized_queue)

            # Normalize total speed
            lane_total_speed = sum(
                vehicle["speed"] for vehicle in lane_data["vehicles"]
            )
            normalized_speed = (lane_total_speed - speed_mean) / speed_std
            lane_features.append(normalized_speed)

            # Clean up temporary variables
            del normalized_approach, normalized_wait_time, normalized_queue, lane_total_speed, normalized_speed

            # Add to signal observation
            signal_state.append(lane_features)

        normalized_states[signal_id] = np.expand_dims(np.array(signal_state), axis=0)
        del signal_state, active_phase_index

    # Remove temporary variables
    del approach_mean, approach_std, wait_time_mean, wait_time_std, queue_mean, queue_std, speed_mean, speed_std

    return normalized_states
