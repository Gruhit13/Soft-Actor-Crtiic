from networks import SoftQNetwork, PolicyNetwork
from buffer import ReplayBuffer

import torch as T
from torch import optim
from torch.nn import functional as F
from torch.distributions import Normal

import numpy as np

from gym import Env

class Agent:
    def __init__(self, env: Env, gamma: float, tau: float, alpha: float,
        q_lr: float,  policy_lr: float, a_lr:float, buffer_maxlen: int):

        self.env = env
        self.action_range = [self.env.action_space.low[0], self.env.action_space.high[0]]
        self.obs_dim = self.env.observation_space.shape
        self.n_action = self.env.action_space.shape[0]

        self.gamma = gamma
        self.tau = tau
        self.update_step = 0
        self.delay_step = 2

        self.q_net1 = SoftQNetwork(self.n_action, self.obs_dim)
        self.q_net2 = SoftQNetwork(self.n_action, self.obs_dim)
        self.target_net_1 = SoftQNetwork(self.n_action, self.obs_dim)
        self.target_net_2 = SoftQNetwork(self.n_action, self.obs_dim)
        self.policy_net = PolicyNetwork(self.n_action, self.obs_dim)

        # copy main network parameters to target network
        for target_params, param in zip(self.target_net_1.parameters(), self.q_net1.parameters()):
            target_params.data.copy_((1-self.tau)*target_params + self.tau*param)
        
        for target_params, param in zip(self.target_net_2.parameters(), self.q_net2.parameters()):
            target_params.data.copy_((1-self.tau)*target_params + self.tau*param)
        
        # Initializer Optimizer
        self.q1_optimizer = optim.Adam(self.q_net1.parameters(), lr=q_lr)
        self.q2_optimizer = optim.Adam(self.q_net2.parameters(), lr=q_lr)
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=policy_lr)  

        # Initialize Entropy
        self.alpha = alpha
        self.target_entropy = -T.prod(T.Tensor(self.env.action_space.shape)).item()
        self.log_alpha = T.zeros(1, requires_grad=True)
        self.alpha_optim = optim.Adam([self.log_alpha], lr=a_lr)

        self.replay_buffer = ReplayBuffer(self.obs_dim, self.n_action, buffer_maxlen)
    
    def get_action(self, state: np.ndarray):
        state = T.FloatTensor(state).unsqueeze(0)
        mean, log_std = self.policy_net(state)
        std = log_std.exp()

        normal = Normal(mean, std)
        z = normal.sample()
        action = T.tanh(z)
        action = action.detach().cpu().squeeze(0).numpy()

        return self.rescale_action(action)

    def rescale_action(self, action: np.ndarray):
        return action * (self.action_range[1] - self.action_range[0]) / 2.0 + \
            (self.action_range[1] - self.action_range[0]) / 2.0
        
    def append(self, s: np.ndarray, a: np.ndarray, r: float, d: bool, n_s: np.ndarray):
        self.replay_buffer.store_experience(s, a, r, d, n_s)
    
    def update(self, batch_size: int):

        if batch_size > self.replay_buffer.max_buffer:
            return

        states, actions, rewards, dones, next_states = self.replay_buffer.sample(batch_size)
        
        states = T.FloatTensor(states)
        actions = T.FloatTensor(actions)
        rewards = T.FloatTensor(rewards)
        dones = T.FloatTensor(dones)
        next_states = T.FloatTensor(next_states)

        next_actions, next_log_pi = self.policy_net.sample_action(next_states)
        next_q1 = self.q_net1(next_states, next_actions)
        next_q2 = self.q_net2(next_states, next_actions)
        next_q_target = T.min(next_q1, next_q2) - self.alpha * next_log_pi
        expected_q = rewards + ((1 - dones) * self.gamma * next_q_target)

        # Q Loss
        curr_q1 = self.q_net1(states, actions)
        curr_q2 = self.q_net2(states, actions)
        q1_loss = F.mse_loss(curr_q1, expected_q.detach())
        q2_loss = F.mse_loss(curr_q2, expected_q.detach())

        # Update q networks
        self.q1_optimizer.zero_grad()
        q1_loss.backward()
        self.q1_optimizer.step()

        self.q2_optimizer.zero_grad()
        q2_loss.backward()
        self.q2_optimizer.step()

        # Delayedly update the policy network
        new_actions, log_pi = self.policy_net(states)
        if self.update_step % self.delay_step == 0:
            min_q = T.min(
                self.q_net1.forward(states, new_actions),
                self.q_net2.forward(states, new_actions)
            )
            
            policy_loss = (self.alpha * log_pi - min_q).mean()

            self.policy_optimizer.zero_grad()
            policy_loss.backward()
            self.policy_optimizer.step()

            # Target network Update
            for target_params, param in zip(self.target_net_1.parameters(), self.q_net1.parameters()):
                target_params.data.copy_((1-self.tau)*target_params + self.tau*param)
            
            for target_params, param in zip(self.target_net_2.parameters(), self.q_net2.parameters()):
                target_params.data.copy_((1-self.tau)*target_params + self.tau*param)
            
        
        # Update Temperature
        alpha_loss = (self.log_alpha * (-log_pi - self.target_entropy).detach()).mean()

        self.alpha_optim.zero_grad()
        alpha_loss.backward()
        self.alpha_optim.step()

        self.update_step += 1