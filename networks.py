import torch as T
from torch import nn
import torch.nn.functional as F
from torch.distributions import Normal
from typing import Tuple

class SoftQNetwork(nn.Module):
    def __init__(self, n_action: int, obs_shape: Tuple, hidden_dim : int=256, w_init: float=3e-3):
        super(SoftQNetwork, self).__init__()
        self.linear1 = nn.Linear(obs_shape[-1] + n_action, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, 1)

        self.linear3.weight.data.uniform_(-w_init, w_init)
        self.linear3.bias.data.uniform_(-w_init, w_init)
    
    def forward(self, state: T.tensor, action: T.Tensor) -> T.Tensor:
        x = T.concat([state, action], dim=-1)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))

        q_val = self.linear3(x)

        return q_val

class PolicyNetwork(nn.Module):
    def __init__(self, n_action: int, obs_shape: Tuple, log_std_min: float = -20,
        log_std_max: float = 2, hidden_dim: int=256, w_init: float = 3e-3):
        
        super(PolicyNetwork, self).__init__()

        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        self.linear1 = nn.Linear(obs_shape[-1], hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        
        self.mean_linear = nn.Linear(hidden_dim, n_action)
        self.mean_linear.weight.data.uniform_(-w_init, w_init)
        self.mean_linear.bias.data.uniform_(-w_init, w_init)

        self.log_std_linear = nn.Linear(hidden_dim, n_action)
        self.log_std_linear.weight.data.uniform_(-w_init, w_init)
        self.log_std_linear.bias.data.uniform_(-w_init, w_init)

    def forward(self, state: T.Tensor) -> Tuple[T.Tensor, T.Tensor]:
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))

        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = T.clamp(log_std, self.log_std_min, self.log_std_max)
        return mean, log_std
    
    def sample_action(self, state: T.Tensor, epsilon = 1e-6):
        mean, log_std = self.forward(state)
        std = log_std.exp()

        normal = Normal(mean, std)
        z = normal.rsample()
        action = T.tanh(z)

        log_pi = normal.log_prob(z) - T.log(1 - action.pow(2) + epsilon)
        log_pi = log_pi.sum(1, keepdim=True)

        return action, log_pi