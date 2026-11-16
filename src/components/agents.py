import torch
import torch.nn as nn
import numpy as np
from src.utils.simulation_utils import get_traci

from dataclasses import dataclass, field
from typing import Dict, Tuple, Any
from pathlib import Path
from pfrl.agents import PPO
from pfrl.policies import SoftmaxCategoricalHead


def initialize_device():
    """
    Initialize the optimal device for tensor operations (GPU or CPU)
    pfrl.agents does not support MPS
    """
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    else:
        return torch.device("cpu")


device = initialize_device()


class BiLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, action_size, dropout=0.5):
        super(BiLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Define the BiLSTM layer
        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout,
        )

        # Define the policy and value heads
        self.policy_head = nn.Sequential(nn.Linear(hidden_size * 2, action_size), SoftmaxCategoricalHead())
        self.value_head = nn.Linear(hidden_size * 2, 1)

    def forward(self, x):
        # Reshape input to (batch_size, sequence_length, input_size)
        batch_size = x.size(0)
        x = x.view(batch_size, -1, x.size(1) * x.size(2) * x.size(3))

        # Initialize hidden state and cell state
        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(x.device)  # *2 for bidirectional
        c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(x.device)

        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))

        # Decode the hidden state of the last time step
        out = out[:, -1, :]
        action_distrib = self.policy_head(out)
        state_value = self.value_head(out)

        # Remove temporary variables
        del h0, c0, out

        return action_distrib, state_value


@dataclass
class PPOAgentController:
    """
    PPOAgentController class that manages multiple traffic signal control agents.
    """

    episodes: int
    obs_act: Dict[str, Tuple]
    sim_config: Any
    training: bool = True
    _models: Dict[str, nn.Module] = field(default_factory=dict)
    _optimizers: Dict[str, torch.optim.Adam] = field(default_factory=dict)
    _ppo_agents: Dict[str, PPO] = field(default_factory=dict)

    def __post_init__(self):
        """
        Initializes PPO agents for each traffic signal. Each agent has its own neural network and learns independently based on its local observations.
        """
        print(f"Using device: {device}")

        # Create a separate PPO agent for each traffic signal
        for signal_id, (observation_space, action_space) in self.obs_act.items():
            input_size = (observation_space[0] * observation_space[1] * observation_space[2])
            hidden_size = 128
            num_layers = 4
            output_size = action_space

            # Build the BiLSTM model for the signal
            self._models[signal_id] = BiLSTM(input_size, hidden_size, num_layers, output_size)

            # Initialize biases to zero
            for module in self._models[signal_id].modules():
                if isinstance(module, (nn.Linear, nn.Conv2d)):
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)

            # Move model to the appropriate device
            self._models[signal_id] = self._models[signal_id].to(device)

            # Initialize the optimizer for neural network training
            self._optimizers[signal_id] = torch.optim.Adam(
                self._models[signal_id].parameters(),
                lr=self.sim_config.hyperparameters.learning_rate,
                eps=self.sim_config.hyperparameters.optimizer_epsilon,
            )

            # Create the PPO agent with the model and optimizer
            self._ppo_agents[signal_id] = PPO(
                self._models[signal_id],
                self._optimizers[signal_id],
                gpu=device.index,
                phi=lambda x: torch.tensor(np.asarray(x, dtype=np.float32), device=device),
                clip_eps=self.sim_config.hyperparameters.policy_clip_range,
                clip_eps_vf=self.sim_config.hyperparameters.value_clip_range,
                update_interval=self.sim_config.hyperparameters.update_interval,
                minibatch_size=self.sim_config.hyperparameters.minibatch_size,
                epochs=self.sim_config.hyperparameters.ppo_epochs,
                standardize_advantages=self.sim_config.hyperparameters.standardize_advantages,
                entropy_coef=self.sim_config.hyperparameters.entropy_coefficient,
                max_grad_norm=self.sim_config.hyperparameters.max_gradient_norm,
            )

            # Remove temporary variables
            del input_size, hidden_size, num_layers, output_size

    def act(self, observation: Dict[str, Any]) -> Dict[str, int]:
        """
        Select actions for each traffic signal based on their observations.
        """
        actions = {}

        for signal_id, obs in observation.items():
            # Use evaluation mode during testing for deterministic behavior
            if not self.training:
                with self._ppo_agents[signal_id].eval_mode():
                    actions[signal_id] = self._ppo_agents[signal_id].act(obs)
            else:
                # Use exploration during training for better policy learning
                actions[signal_id] = self._ppo_agents[signal_id].act(obs)

        return actions

    def observe(self, observation: Dict[str, Any], reward: Dict[str, float], done: bool) -> None:
        """
        Observes the current state of the environment and rewardss.
        """
        for signal_id, obs in observation.items():
            self._ppo_agents[signal_id].observe(obs, reward[signal_id], done, False)

    def save(self, base_path: Path) -> None:
        """
        Save each traffic signal PPO agent model
        """
        base_path.mkdir(parents=True, exist_ok=True)

        for signal_id in self._ppo_agents.keys():
            model_filename = f"model_{signal_id}.pt"
            model_path = base_path / model_filename

            torch.save(
                {
                    "model_state_dict": self._models[signal_id].state_dict(),
                    "optimizer_state_dict": self._optimizers[signal_id].state_dict(),
                },
                model_path,
            )

    def load(self, base_path: Path) -> None:
        """
        Load each traffic signal PPO agent model.
        """
        # Load each agent from its corresponding file
        for signal_id in self._ppo_agents.keys():
            model_filename = f"model_{signal_id}.pt"
            model_path = Path(base_path / model_filename)

            # Load to the appropriate device
            map_location = device
            checkpoint = torch.load(model_path, map_location=map_location, weights_only=True)
            self._models[signal_id].load_state_dict(checkpoint["model_state_dict"])
            self._optimizers[signal_id].load_state_dict(checkpoint["optimizer_state_dict"])

        self.training = False


@dataclass
class FixedIntervalAgentController:
    """
    Non-learning agent that follows a simple round-robin phase selection.
    """

    training: bool = False

    def act(self, observation):
        """
        Select the next traffic signal phase in sequence for each intersection.
        """
        # Get traci when needed
        traci = get_traci()

        actions = {}
        for signal_id, _ in observation.items():
            # Get the current phase index for this traffic signal
            current_phase = traci.trafficlight.getPhase(signal_id)
            # Calculate the next phase index (cycling through available phases)
            total_phases = len(traci.trafficlight.getAllProgramLogics(signal_id)[0].phases)
            next_phase = (current_phase + 1) % total_phases
            actions[signal_id] = next_phase
        return actions

    def observe(self, observation, reward, done):
        pass

    def save(self, path: Path) -> None:
        pass

    def load(self, path: Path) -> None:
        pass
