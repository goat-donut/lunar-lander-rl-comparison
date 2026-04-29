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
        self.l2 = nn.Linear(128, 128)
        self.l3 = nn.Linear(128, action_dim)
        self.relu = nn.ReLU()
    def forward(self, x):
        x = self.l1(x)
        x = self.relu(x)
        x = self.l2(x)
        x = self.relu(x)
        x = self.l3(x)
        return x
    

class ReplayBuffer:
    def __init__(self, capacity=100000, batch_size=64):
        self.capacity = capacity
        self.buffer = deque(maxlen=self.capacity)
        self.batch_size = batch_size
    
    def __len__(self):
        return len(self.buffer)
    
    def store(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self):
        batch = random.sample(self.buffer, self.batch_size)
        states = [sample[0] for sample in batch]
        actions = [sample[1] for sample in batch]
        rewards = [sample[2] for sample in batch]
        next_states = [sample[3] for sample in batch]
        done_list = [sample[4] for sample in batch]
        return states, actions, rewards, next_states, done_list

class DQN:
    def __init__(self, state_dim, action_dim, name = 'DQN'):
        self.name = name
        self.device = torch.device("cpu")
        self.q_net = Net(state_dim, action_dim).to(self.device)
        self.target_net = Net(state_dim, action_dim).to(self.device)
        self.buffer = ReplayBuffer()
        self.eps = 1.0
        self.min_eps = 0.01
        self.eps_decay = 0.999
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.gamma = 0.99   
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=5e-4)
        self.t = 0
    def select_action(self, state):
        if random.random() < self.eps:
            action = random.randint(0, self.action_dim - 1)
            return action
        else:
            state = torch.FloatTensor(np.array(state)).unsqueeze(0).to(self.device)
            q_values = self.q_net(state)
            best_action = q_values.argmax().item()
            return best_action
    
    def update_eps(self):
        self.eps = max(self.eps * self.eps_decay, self.min_eps)
    def update_target(self):
        self.target_net.load_state_dict(self.q_net.state_dict())
    def train(self):
        if len(self.buffer) >= 1000:
            states, actions, rewards, next_states, done_list= self.buffer.sample()
            
            
            states = torch.FloatTensor(np.array(states)).to(self.device)
            actions = torch.LongTensor(np.array(actions)).unsqueeze(1).to(self.device)  # gather
            rewards = torch.FloatTensor(np.array(rewards)).to(self.device)
            next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
            done_list = torch.FloatTensor(np.array(done_list)).to(self.device)
            
            all_q_pred = self.q_net(states)
            q_pred = all_q_pred.gather(1, actions)
            
            
            with torch.no_grad():
                next_q = self.target_net(next_states).max(1)[0]
                
            target = rewards + self.gamma * next_q * (1 - done_list)
            
            loss = F.mse_loss(q_pred, target.unsqueeze(1))
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            self.t += 1
            if self.t % 500 == 0:
                self.update_target()
        