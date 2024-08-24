import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Callable, Tuple
from models import GaussianNetwork, QNetworkSep, DeterministicNetwork, GaussianNetwork
from utils.misc import hard_update
from torch.optim.adam import Adam


class SAC(nn.Module):

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
        alpha=0.1,
        gamma=0.99,
        tau=0.005,
        update_interval=1,
        policy_type="gaussian",
    ) -> None:
        super(SAC, self).__init__()

        self.action_dim = action_dim
        self.state_dim = state_dim

        self.n_hidden_layers = n_hidden_layers
        self.hidden_dim = hidden_dim

        self.critic = QNetworkSep(
            state_dim + action_dim, 1, n_hidden_layers, hidden_dim, activation_func
        )
        self.critic_optim = Adam(self.critic.parameters(), lr=lr)

        self.critic_target = QNetworkSep(
            state_dim + action_dim, 1, n_hidden_layers, hidden_dim, activation_func
        )

        self.scale = scale
        self.bias = bias
        self.alpha = alpha
        self.gamma = gamma
        self.tau = tau
        self.update_interval = update_interval
        self.episode = 0
        if policy_type.lower() == "gaussian":
            self.policy = GaussianNetwork(
                state_dim,
                action_dim,
                n_hidden_layers,
                n_hidden_layers,
                hidden_dim,
                scale,
                bias,
                activation_func,
            )
        else:
            self.alpha = 0.0
            self.policy = DeterministicNetwork(
                state_dim,
                action_dim,
                n_hidden_layers,
                n_hidden_layers,
                hidden_dim,
                scale,
                bias,
                activation_func,
            )
        self.policy_optim = Adam(self.policy.parameters(), lr)
        # weights_init_(self)
        hard_update(self.critic, self.critic_target)
        return

    def sample(
        self, state: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.policy.sample(state)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        action, _, _ = self.policy.sample(state)
        return action


class DiscreteSAC(nn.Module):
    def __init__(self, *args, n_classes=2, **kwargs):
        super(DiscreteSAC, self).__init__()
        self.model = SAC(*args, **kwargs)
        self.n_classes = n_classes
        return

    def sample(
        self, state: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.model.sample(state)

    def discrete_sample(
        self, state: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        action, b, c = self.model.policy.sample(state)
        a = F.one_hot(torch.argmax(action), num_classes=self.n_classes)
        return a, b, c

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        action, _, _ = self.discrete_sample(state)
        return action
