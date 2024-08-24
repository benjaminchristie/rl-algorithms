import torch
import torch.nn as nn
from torch.distributions import Normal
from typing import Callable, Tuple


LOG_SIG_MAX = 2
LOG_SIG_MIN = -20
EPSILON = 1e-6


class QNetworkSep(nn.Module):

    def __init__(
        self,
        n_inputs: int,
        n_outputs: int,
        n_hidden_layers: int,
        hidden_dim: int,
        activation_func: Callable[[torch.Tensor], torch.Tensor] | None,
    ) -> None:
        super(QNetworkSep, self).__init__()
        if activation_func is None:
            self.activation_func = nn.Identity()
        else:
            self.activation_func = activation_func
        self.layers_1 = nn.ParameterList(
            [nn.Linear(n_inputs, hidden_dim)]
            + [nn.Linear(hidden_dim, hidden_dim) for _ in range(n_hidden_layers - 1)]
            + [nn.Linear(hidden_dim, n_outputs)]
        )
        self.layers_2 = nn.ParameterList(
            [nn.Linear(n_inputs, hidden_dim)]
            + [nn.Linear(hidden_dim, hidden_dim) for _ in range(n_hidden_layers - 1)]
            + [nn.Linear(hidden_dim, n_outputs)]
        )
        self.n_layers = n_hidden_layers + 1

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        does not compute activation_func on output of last layer
        """
        x1 = x
        x2 = x
        for i in range(self.n_layers - 1):
            x1 = self.layers_1[i](x1)
            x1 = self.activation_func(x1)
        x1 = self.layers_1[self.n_layers - 1](x1)
        for i in range(self.n_layers - 1):
            x2 = self.layers_2[i](x2)
            x2 = self.activation_func(x2)
        x2 = self.layers_2[self.n_layers - 1](x2)
        return x1, x2


class QNetwork(nn.Module):

    def __init__(
        self,
        n_inputs: int,
        n_outputs: int,
        n_hidden_layers: int,
        hidden_dim: int,
        activation_func: Callable[[torch.Tensor], torch.Tensor] | None,
    ) -> None:
        super(QNetwork, self).__init__()
        if activation_func is None:
            self.activation_func = nn.Identity()
        else:
            self.activation_func = activation_func
        self.layers = nn.ParameterList(
            [nn.Linear(n_inputs, hidden_dim)]
            + [nn.Linear(hidden_dim, hidden_dim) for _ in range(n_hidden_layers - 1)]
            + [nn.Linear(hidden_dim, n_outputs)]
        )
        self.n_layers = n_hidden_layers + 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        does not compute activation_func on output of last layer
        """
        for i in range(self.n_layers - 1):
            x = self.layers[i](x)
            x = self.activation_func(x)
        x = self.layers[self.n_layers - 1](x)
        return x


class DeterministicNetwork(nn.Module):
    def __init__(
        self,
        n_inputs: int,
        n_outputs: int,
        n_hidden_layers: int,
        n_normal_layers: int,
        hidden_dim: int,
        scale: torch.Tensor | float,
        bias: torch.Tensor | float,
        activation_func: Callable[[torch.Tensor], torch.Tensor] | None,
    ):
        super(DeterministicNetwork, self).__init__()
        self.scale = scale
        self.bias = bias
        self.hidden_dim = hidden_dim
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.n_hidden_layers = n_hidden_layers
        self.n_normal_layers = n_normal_layers
        if activation_func is None:
            self.activation_func = nn.Identity()
        else:
            self.activation_func = activation_func
        self.layers = nn.ParameterList(
            [nn.Linear(n_inputs, hidden_dim)]
            + [nn.Linear(hidden_dim, hidden_dim) for _ in range(n_hidden_layers - 1)]
        )
        self.mean_layers = nn.ParameterList(
            [nn.Linear(hidden_dim, hidden_dim) for _ in range(n_normal_layers - 1)]
            + [nn.Linear(hidden_dim, n_outputs)]
        )
        return

    def sample(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, _ = self.forward(x)
        y_t = self.scale * torch.sigmoid(mu) + self.bias
        return y_t, torch.FloatTensor([0.0]), y_t

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        for i in range(self.n_hidden_layers):
            x = self.activation_func(self.layers[i](x))
        mu = x
        for i in range(self.n_normal_layers - 1):
            mu = self.activation_func(self.mean_layers[i](mu))
        mu = self.mean_layers[self.n_normal_layers - 1](mu)
        return mu, 0.0 * mu


class GaussianNetwork(nn.Module):
    def __init__(
        self,
        n_inputs: int,
        n_outputs: int,
        n_hidden_layers: int,
        n_normal_layers: int,
        hidden_dim: int,
        scale: torch.Tensor | float,
        bias: torch.Tensor | float,
        activation_func: Callable[[torch.Tensor], torch.Tensor] | None,
    ):
        super(GaussianNetwork, self).__init__()
        self.scale = scale
        self.bias = bias
        self.hidden_dim = hidden_dim
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.n_hidden_layers = n_hidden_layers
        self.n_normal_layers = n_normal_layers
        if activation_func is None:
            self.activation_func = nn.Identity()
        else:
            self.activation_func = activation_func
        self.layers = nn.ParameterList(
            [nn.Linear(n_inputs, hidden_dim)]
            + [nn.Linear(hidden_dim, hidden_dim) for _ in range(n_hidden_layers - 1)]
        )
        self.mean_layers = nn.ParameterList(
            [nn.Linear(hidden_dim, hidden_dim) for _ in range(n_normal_layers - 1)]
            + [nn.Linear(hidden_dim, n_outputs)]
        )
        self.std_layers = nn.ParameterList(
            [nn.Linear(hidden_dim, hidden_dim) for _ in range(n_normal_layers - 1)]
            + [nn.Linear(hidden_dim, n_outputs)]
        )
        return

    def sample(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, log_std = self.forward(x)
        std = log_std.exp()
        normal = Normal(mu, std)
        x_t = normal.rsample()
        y_t = torch.sigmoid(x_t)
        log_prob = normal.log_prob(x_t)
        # enforce bounds
        log_prob -= torch.log(self.scale * (1 - y_t.pow(2)) + EPSILON)
        log_prob = log_prob.sum(1, keepdim=True)
        mu = self.scale * torch.sigmoid(mu) + self.bias
        action = y_t * self.scale + self.bias
        return action, log_prob, mu

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        for i in range(self.n_hidden_layers):
            x = self.activation_func(self.layers[i](x))
        mu = x
        std = x
        for i in range(self.n_normal_layers - 1):
            mu = self.activation_func(self.mean_layers[i](mu))
        mu = self.mean_layers[self.n_normal_layers - 1](mu)
        for i in range(self.n_normal_layers - 1):
            std = self.activation_func(self.std_layers[i](std))
        log_std = self.std_layers[self.n_normal_layers - 1](std)
        log_std = torch.clamp(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
        return mu, log_std
