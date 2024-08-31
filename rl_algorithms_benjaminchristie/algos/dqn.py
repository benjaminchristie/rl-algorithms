import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Callable, Tuple
from rl_algorithms_benjaminchristie.algos.models import (
    GaussianNetwork,
    DeterministicNetwork,
    GaussianNetwork,
)
from rl_algorithms_benjaminchristie.utils.misc import hard_update
from torch.optim.adam import Adam


class DQN(nn.Module):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        n_hidden_layers: int,
        hidden_dim: int,
        activation_func: Callable[[torch.Tensor], torch.Tensor] | None,
        scale: float,
        bias: float,
        lr: float,
        gamma=0.99,
        epsilon=0.01,
        tau=0.005,
        policy_type="deterministic",
    ) -> None:
        super(DQN, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.n_hidden_layers = n_hidden_layers
        self.hidden_dim = hidden_dim
        self.activation_func = activation_func
        self.scale = scale
        self.bias = bias
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.tau = tau

        if policy_type.lower() == "gaussian":

            self.eval_net = GaussianNetwork(
                state_dim,
                action_dim,
                n_hidden_layers // 2,
                n_hidden_layers // 2,
                hidden_dim,
                scale,
                bias,
                activation_func,
            )

            self.target_net = GaussianNetwork(
                state_dim,
                action_dim,
                n_hidden_layers // 2,
                n_hidden_layers // 2,
                hidden_dim,
                scale,
                bias,
                activation_func,
            )

        else:
            self.eval_net = DeterministicNetwork(
                state_dim,
                action_dim,
                n_hidden_layers // 2,
                n_hidden_layers // 2,
                hidden_dim,
                scale,
                bias,
                activation_func,
            )

            self.target_net = DeterministicNetwork(
                state_dim,
                action_dim,
                n_hidden_layers // 2,
                n_hidden_layers // 2,
                hidden_dim,
                scale,
                bias,
                activation_func,
            )

        self.optim = Adam(self.eval_net.parameters(), lr)

    def sample(
        self, state: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.eval_net.sample(state)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        action, _, _ = self.sample(state)
        action_hot = F.one_hot(torch.argmax(action, dim=1), num_classes=self.action_dim)
        return action_hot

    def select_action(self, state: torch.Tensor, epsilon=-1.0) -> torch.Tensor:
        if epsilon == -1:
            epsilon = self.epsilon
        c = np.random.rand()
        if c < epsilon:
            action_value = torch.rand(self.action_dim)
        else:
            action_value, _, _ = self.eval_net.sample(state)
        action = torch.argmax(action_value)
        return action
