from dataclasses import field, dataclass
from typing import Any, Dict, List, Optional, Set, Tuple

import copy
from src.utils.simulation_utils import get_traci


@dataclass
class PhaseManager:
    """
    PhaseManager is used to keep track of various phases for the traffic signal
    and create phases for the yellow light, which will enable progression of
    phases.
    """
    phases: List[Any]
    yellow_light_duration: int
    yellow_dict: Dict[str, int] = field(default_factory=dict)  # Maps transition keys to yellow phase indices

    def generate_yellow_phases(self) -> Tuple[List[Any], Dict[str, int]]:
        """
        Change the phases from red to green, by creating the yellow phase, by
        exhaustive search.
        """
        # Get traci when needed
        traci = get_traci()

        # Copying the phases to avoid reference modification
        phases_copy = copy.copy(self.phases)

        # Exhaustive search for phase
        for i, phase_from in enumerate(self.phases):
            for j, phase_to in enumerate(self.phases):
                if i != j:  # Skip self-transitions
                    need_yellow = False
                    yellow_str = ""

                    # Checking each signal phase
                    for sig_idx in range(len(phase_from.state)):
                        s_from = phase_from.state[sig_idx]
                        s_to = phase_to.state[sig_idx]

                        # Checking when transition is happening between red and green
                        if (s_from == "G" or s_from == "g") and (
                            s_to == "r" or s_to == "s"
                        ):
                            need_yellow = True
                            yellow_str += "y"
                        else:
                            yellow_str += s_from

                    # When the phase is known and needed, add to the phases list
                    if need_yellow:
                        phases_copy.append(
                            traci.trafficlight.Phase(
                                self.yellow_light_duration, yellow_str
                            )
                        )

                        # Keep track of the yellow phase indices for mapping.
                        self.yellow_dict[f"{i}_{j}"] = len(phases_copy) - 1

        return phases_copy, self.yellow_dict

@dataclass
class LaneManager:
    """
    Class to keep track the lanes for directions and the connections.
    """
    signal_id: str
    signal_config: Dict = field(default_factory=dict)

    # Lanes vars
    lanes: List[str] = field(default_factory=list, init=False)  # inbound
    outbound_lanes: List[str] = field(default_factory=list, init=False)  # outbound

    # mapping for lanes
    lane_sets: Dict[str, List[str]] = field(default_factory=dict, init=False)  # inbound map
    lane_sets_outbound: Dict[str, List[str]] = field(default_factory=dict, init=False)  # outbound map
    in_from_direction: Dict[str, List[str]] = field(default_factory=dict, init=False)  # direction-lanes mapping

    # connectivity for lanes for downstream
    downstream: Dict[str, Optional[str]] = field(default_factory=dict, init=False)
    out_lane_to_signalid: Dict[str, str] = field(default_factory=dict, init=False)


    def initialize(self, signal_id: str) -> None:
        """
        Setup and configure the lane manager for a specific signal (intersection)
        """
        # skip if not found
        if signal_id not in self.signal_config:
            return

        # get configs
        config = self.signal_config[signal_id]
        self.lane_sets = config["lane_sets"]
        self.lane_sets_outbound = {i: [] for i in self.lane_sets}
        self.downstream = config["downstream"]

        # compute the lanes and mapping
        self.get_lane_sets()
        self.get_downstream_lanes()

    def get_lane_sets(self) -> None:
        """
        Map the lanes based of directions for an overview, for a signal.
        """

        reverse_direction_dict = {"E": "W", "N": "S", "S": "N", "W": "E"}

        # for each lane
        for direction, lanes in self.lane_sets.items():
            # other direction
            inbound_dir = reverse_direction_dict[direction.split("-")[0]]

            for lane in lanes: # for each lane
                if lane not in self.lanes:
                    self.lanes.append(lane)

                if inbound_dir not in self.in_from_direction:
                    self.in_from_direction[inbound_dir] = [] # init

                # add lane to direction mapping if not known before
                if lane not in self.in_from_direction[inbound_dir]:
                    self.in_from_direction[inbound_dir].append(lane)

        # Remove temporary variables
        del reverse_direction_dict

    def get_downstream_lanes(self) -> None:
        """
        get the downstream lanes for the map.
        """
        for direction, downstream_signal in self.downstream.items():
            if downstream_signal is None:
                continue

            # get lanes from the config for singal
            downstream_lane_sets = self.signal_config[downstream_signal]["lane_sets"]

            # for each downstream lanes set
            for dir, downstream_lane_set in downstream_lane_sets.items():
                direction_split = dir.split("-")[0]
                if direction_split == direction:
                    # for each lane
                    for lane in downstream_lane_set:
                        # when lane not found, add
                        if lane not in self.outbound_lanes:
                            self.outbound_lanes.append(lane)

                        # lane to downstream signal mapping
                        self.out_lane_to_signalid[lane] = downstream_signal

                       # find the correct outbound lane set
                        for lane_set in self.lane_sets:
                            lane_set_split = lane_set.split("-")[1]
                            if direction_split == lane_set_split:
                                if lane not in self.lane_sets_outbound[lane_set]:
                                    self.lane_sets_outbound[lane_set].append(lane)

        # Remove temporary variables after processing each direction


class Signal:
    """
    Class to maintain the traffic signal and its components.
    """

    def __init__(
        self,
        signal_config: dict,
        sumo: Any,
        signal_id: str,
        yellow_light_duration: int,
        phases: List[Any],
    ):
        """
        Components for the signal
        """
        self.sumo = sumo  # SUMO connection for traffic control
        self.id = signal_id  # Unique identifier for this signal
        self.next_phase = 0  # Next traffic light phase to apply

        # managers
        self.phase_manager = PhaseManager(phases, yellow_light_duration)
        self.lane_manager = LaneManager(signal_id, signal_config)
        self.lane_manager.initialize(signal_id)
        if signal_id not in signal_config:
            self.lane_manager.lanes = list(dict.fromkeys(sumo.trafficlight.getControlledLanes(signal_id)))

        # keep track of observations and metrics
        self.waiting_times: Dict[str, float] = {}  # use vehicle id
        self.signals = None
        self.full_observation = None
        self.last_step_vehicles = None

        # Init traffic light logic with phases
        self.initialize_traffic_signals()

    @property
    def lanes(self) -> List[str]:
        # get from the lane manager
        return self.lane_manager.lanes

    def initialize_traffic_signals(self) -> None:
        """
        Construct the traffic signal phase logic and configure the yellow lights
        phases for SUMO
        """
        # Get the yellow phases.
        self.phases, self.yellow_dict = self.phase_manager.generate_yellow_phases()

        # From SUMO, get the traffic signal logic for the intersection, which has phases.
        program_logics = self.sumo.trafficlight.getAllProgramLogics(self.id)
        # Generally there is only one logic that contains the phases
        logic = program_logics[0]

        # default
        logic.type = 0
        # set using the phases generated
        logic.phases = self.phases
        # set the logic back with the new phases
        self.sumo.trafficlight.setProgramLogic(self.id, logic)

    def observe(self, step_length: int, distance: int) -> None:
        """
        Observe the SUMO environment and collect metrics.
        """
        observations = {}
        vehicles = set()

        # For each lane, observe
        for lane in self.lane_manager.lanes:
            lane_metrics = self.get_lane_metrics(lane, step_length, distance)
            observations[lane] = lane_metrics

            # get the vehicles in the lanes
            for vehicle in lane_metrics["vehicles"]:
                vehicles.add(vehicle["id"])

        self.update_vehicle_info(vehicles, observations)

        # keep track of observations
        self.last_step_vehicles = vehicles
        self.full_observation = observations

        # Remove temporary variables
        del observations, vehicles

    @property
    def phase(self) -> int:
        """
        get the current phase from SUMO env
        """
        return self.sumo.trafficlight.getPhase(self.id)

    def prep_phase(self, new_phase: int) -> None:
        """
        prepare the phase for the next step
        """
        if self.phase == new_phase:
            # if same, keep as is.
            self.next_phase = self.phase
        else:
            # get and set it to the new phase
            self.next_phase = new_phase

            # is yellow required
            transition_key = f"{self.phase}_{new_phase}"
            if transition_key in self.yellow_dict:
                # Set the appropriate yellow phase
                self.sumo.trafficlight.setPhase(self.id, self.yellow_dict[transition_key])

    def set_phase(self) -> None:
        """
        Set the next phase for the traffic signal.
        """
        self.sumo.trafficlight.setPhase(self.id, int(self.next_phase))

    def get_lane_metrics(self, lane: str, step_length: int, distance: int) -> Dict:
        """
        Get a detailed metrics dictionary for a specific lane.
        """

        # Initialize metrics dictionary
        lane_metrics = {
            "queue": 0,
            "total_wait": 0,
            "approach": 0,
            "vehicles": [],
            "max_wait": 0,
        }

        # Get vehicles in the lane within the specified distance
        vehicles_in_lane = self.get_vehicle_count(distance, lane)
        lane_metrics["vehicles"] = []

        # Process each vehicle in the lane
        for vehicle_id in vehicles_in_lane:
            vehicle_metrics = self.get_vehicle_info(vehicle_id, step_length)
            lane_metrics["vehicles"].append(vehicle_metrics)

            # Update lane-level metrics based on vehicle state
            if vehicle_metrics["wait"] > 0:
                lane_metrics["total_wait"] += vehicle_metrics["wait"]
                lane_metrics["queue"] += 1
                lane_metrics["max_wait"] = max(lane_metrics["max_wait"], vehicle_metrics["wait"])
            else:
                lane_metrics["approach"] += 1
            del vehicle_metrics

        # Remove temporary variables
        del vehicles_in_lane

        return lane_metrics

    def update_vehicle_info(self, vehicles: Set[str], observations: Dict) -> None:
        """
        Update tracking of vehicle information
        """
        if self.last_step_vehicles is None:
            # Update Observations for when it is not the last step
            observations["departures"] = set()
            observations["arrivals"] = vehicles
        else:
            # Update observations for last step
            observations["arrivals"] = vehicles.difference(self.last_step_vehicles)
            observations["departures"] = self.last_step_vehicles.difference(vehicles)

            for vehicle_id in observations["departures"]:
                if vehicle_id in self.waiting_times:
                    self.waiting_times.pop(vehicle_id)

        observations["num_vehicles"] = vehicles

    def get_vehicle_info(self, vehicle: str, step_length: int) -> Dict:
        """
        Collect detailed metrics for a given vehicle
        """
        # Update waiting time vehicles being tracked
        if vehicle in self.waiting_times:
            self.waiting_times[vehicle] += step_length
        elif self.sumo.vehicle.getWaitingTime(vehicle) > 0:
            self.waiting_times[vehicle] = self.sumo.vehicle.getWaitingTime(vehicle)

        return {
            "id": vehicle,
            "acceleration": self.sumo.vehicle.getAcceleration(vehicle),
            "wait": self.waiting_times.get(vehicle, 0),
            "speed": self.sumo.vehicle.getSpeed(vehicle),
            "type": self.sumo.vehicle.getTypeID(vehicle),
            "position": self.sumo.vehicle.getLanePosition(vehicle),
        }

    def get_vehicle_count(self, distance: int, lane: str) -> List[str]:
        """
        Get vehicles on a lane within the specified distance from the intersection.
        """
        # Retrieve the length of the lane and the vehicles on it
        lane_length = self.sumo.lane.getLength(lane)
        vehicles = self.sumo.lane.getLastStepVehicleIDs(lane)

        threshold = lane_length - distance

        # Return vehicles greater than the threshold distance
        return [v for v in vehicles if self.sumo.vehicle.getLanePosition(v) >= threshold]
