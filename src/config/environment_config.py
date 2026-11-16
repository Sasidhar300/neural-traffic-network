from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from src.utils.simulation_utils import start_sumo_connection, close_sumo_connection


@dataclass
class SumoEnvironment:
    network_config : Path = Path("")
    seed : int = -1
    start_time : int = -1
    end_time : int = -1
    connection_name: str = ""
    gui: bool = False
    libsumo: bool = False
    step_ratio: int = -1
    scale: float = -1.0

    _episode_count: int = 0
    _active_signal_count : int = 0
    _connection : Any = None
    _signal_ids : list = field(default_factory=list, init=False)
    _active_signal_ids : list = field(default_factory=list, init=False)
    _signal_phases: dict[str, list] = field(default_factory=dict, init=False)

    def signal_count(self):
        return len(self._signal_ids)

    def active_signal_count(self):
        return len(self._active_signal_ids)

    @property
    def connection(self):
        return self._connection

    @property
    def signal_ids(self):
        return self._signal_ids

    @property
    def active_signal_ids(self):
        return self._active_signal_ids

    @property
    def signal_phases(self):
        return self._signal_phases

    def __post_init__(self):
        self._start(for_initialization=True)
        self._identify_traffic_signals()


    def _start(self, for_initialization=False):
        # Create SUMO command
        use_gui = not for_initialization and self.gui

        # Determine the scale value based on whether this is a training environment
        scales = [0.5, 1.0, 3.0]
        actual_scale = self.scale
        if "train" in self.connection_name and not for_initialization:
            actual_scale_index = int(self._episode_count % 3)
            actual_scale = scales[actual_scale_index]
            if actual_scale == 0: actual_scale = 3.0

        # Use the utility function to start SUMO connection
        self._connection = start_sumo_connection(
            config_path=self.network_config,
            connection_name=self.connection_name if not self.libsumo else "default",
            seed=self.seed,
            start_time=self.start_time,
            end_time=self.end_time,
            scale=actual_scale,
            use_gui=use_gui,
            libsumo=self.libsumo
        )

        # Clean up temporary variables
        del scales, actual_scale

    def _identify_traffic_signals(self):

        # Get traffic signals
        self._signal_ids = self._connection.trafficlight.getIDList()
        self._active_signal_count = self.signal_count()  # Start with all signals active

        # Get valid green phases
        self._signal_phases = {
            signal_id: [
                phase
                for phase in self._connection.trafficlight.getAllProgramLogics(
                    signal_id
                )[0].getPhases()
                if "y" not in phase.state and "g" in phase.state.lower()
            ]
            for signal_id in self._signal_ids
        }

    def close(self):
        # Use the utility function to close SUMO connection
        close_sumo_connection(
            connection_name=self.connection_name if not self.libsumo else None,
            libsumo=self.libsumo
        )

    def step(self):

        for _ in range(self.step_ratio):
            self._connection.simulationStep()

    def reset(self):

        # Close previous simulation
        if self._episode_count != 0:
            self.close()

        self._episode_count += 1
        self._start(for_initialization=False)

        # Setup active signals after reset
        self._setup_active_signals()

    def _setup_active_signals(self):

        if (
            self._episode_count % 30 == 0
            and self._active_signal_count < self.signal_count()
        ):
            self._active_signal_count += 1

        # Select active signals
        self._active_signal_ids[:] = self._signal_ids[:self._active_signal_count]

    def get_time(self):
        return self._connection.simulation.getTime()

    def get_vehicle_ids(self):
        return self._connection.vehicle.getIDList()

    def get_vehicle_emissions(self, vehicle_id):
        return self._connection.vehicle.getCO2Emission(vehicle_id)

    def get_fuel_consumption(self, vehicle_id):
        return self._connection.vehicle.getFuelConsumption(vehicle_id)

    def get_noise_emission(self, vehicle_id):
        return self._connection.vehicle.getNoiseEmission(vehicle_id)
