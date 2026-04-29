import torch 
import torch.nn as nn
import torch.nn.functional as F
from collections import deque
import random
import torch.optim as optim
import numpy as np
class Net(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Net, self).__init__()
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
    
class Reinforce:
    def __init__(self, state_dim, action_dim, gamma, lr, name = 'REINFORCE'):
        self.device = torch.device("cpu")
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.net = Net(self.state_dim, self.action_dim).to(self.device)
        self.optimizer = optim.Adam(self.net.parameters(), lr = lr)
        self.gamma = gamma
        self.max_grad_norm = 0.5
        self.name = name
    def select_action(self, state):
        state = torch.FloatTensor(np.array(state)).unsqueeze(0).to(self.device)
        probs = self.net(state)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action, log_prob
    def compute_return(self, rewards):
        G = 0
        returns = []
        for r in reversed(rewards):
            G = r + self.gamma * G
            returns.insert(0, G)
        return torch.tensor(returns, dtype=torch.float32).to(self.device)
    def train(self, log_probs, returns):
        # normalization
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        
        loss = []
        for log_prob, G in zip(log_probs, returns):
            loss.append(-log_prob * G)
        loss = torch.stack(loss).mean()
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.net.parameters(), self.max_grad_norm)
        self.optimizer.step()