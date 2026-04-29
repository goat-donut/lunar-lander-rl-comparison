import torch 
import torch.nn as nn
import torch.nn.functional as F
from collections import deque
import random
import torch.optim as optim
import numpy as np
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()
        self.l1 = nn.Linear(state_dim, 128)
        self.l2 = nn.Linear(128, 64)
        self.l3 = nn.Linear(64, action_dim)
        self.relu = nn.ReLU()
    def forward(self, x):
        x = self.l1(x)
        x = self.relu(x)
        x = self.l2(x)
        x = self.relu(x)
        x = self.l3(x)
        x = F.softmax(x, dim = -1)
        return x

class Critic(nn.Module):
    def __init__(self, state_dim):
        super(Critic, self).__init__()
        self.state_dim = state_dim
        self.l1 = nn.Linear(self.state_dim, 128)
        self.l2 = nn.Linear(128, 64)
        self.l3 = nn.Linear(64, 1)
    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = self.l3(x)
        return x
    
class ActorCritic:
    def __init__(self, state_dim, action_dim, gamma, lr_actor, lr_critic, name = 'Actor-Critic'):
        self.device = torch.device("cpu")
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.actor = Actor(self.state_dim, self.action_dim).to(self.device)
        self.critic = Critic(self.state_dim).to(self.device)
        self.actor_optim = optim.Adam(self.actor.parameters(), lr = lr_actor)
        self.critic_optim = optim.Adam(self.critic.parameters(), lr = lr_critic)
        self.gamma = gamma
        self.max_grad_norm = 0.5
        self.name = name
    def select_action(self, state):
        state = torch.FloatTensor(np.array(state)).unsqueeze(0).to(self.device)
        probs = self.actor(state)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action, log_prob, dist
    
    def train(self, log_probs, rewards, states, next_states, dones, dists):
        states = torch.FloatTensor(np.array(states)).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        rewards = torch.FloatTensor(np.array(rewards)).to(self.device)
        dones = torch.FloatTensor(np.array(dones)).to(self.device)
        
        V = self.critic(states).squeeze(-1)
        V_next = self.critic(next_states).squeeze().detach()
        
        td_error = rewards + self.gamma * V_next * (1 - dones) - V.detach()
        
        log_probs = torch.stack(log_probs)
        entropy = torch.stack([d.entropy() for d in dists]).mean()
        
        actor_loss = -(log_probs * td_error).mean() - 0.05 * entropy
        target = (rewards + self.gamma * V_next * (1 - dones)).detach()
        critic_loss = F.mse_loss(V, target)
        
        self.actor_optim.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
        self.actor_optim.step()
        
        self.critic_optim.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
        self.critic_optim.step()