from agents.reinforce import Reinforce
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import imageio
import torch
class Trainer:
    def __init__(self, env, agent, episodes, env_name):
        self.env = env
        self.agent = agent
        self.episodes = episodes
        self.env_name = env_name
    
    def record_final(self, n_attempts=5):
        best_frames = None
        best_reward = -np.inf
        
        for _ in range(n_attempts):
            env = gym.make(self.env_name, render_mode='rgb_array')
            state, _ = env.reset()
            frames = []
            total_reward = 0
            
            while True:
                frame = env.render()
                frames.append(frame)
                action, _ = self.agent.select_action(state)
                next_state, reward, terminated, truncated, _ = env.step(action.item())
                done = terminated or truncated
                state = next_state
                total_reward += reward
                if done:
                    break
            
            env.close()
            if total_reward > best_reward:
                best_reward = total_reward
                best_frames = frames
        
        imageio.mimsave(f"result_{self.agent.name}.gif", best_frames, fps=30)
    def train(self):
        reward_history = []
        best_reward = -np.inf
        for ep in range(self.episodes):
            state, _ = self.env.reset()
            rewards = []
            log_probs = []
            total_reward = 0
            while True:
                action, log_prob = self.agent.select_action(state)
                next_state, reward, terminated, truncated, _ = self.env.step(action.item())
                done = terminated or truncated

                state = next_state
                rewards.append(reward)
                log_probs.append(log_prob)
                total_reward += reward
                if done:
                    break
            
            returns = self.agent.compute_return(rewards)
            self.agent.train(log_probs, returns)
            
            reward_history.append(total_reward)
            
            if (ep + 1) >= 200:
                avg_reward = np.mean(reward_history[-200:])
                if avg_reward > best_reward:
                    best_reward = avg_reward
                    torch.save(self.agent.net.state_dict(), f"best_{self.agent.name}.pth")  
                
            if (ep + 1) % 200 == 0:
                avg_reward = np.mean(reward_history[-200:])
                print(f"Episode {ep+1} | Avg reward: {avg_reward:.3f}")
        
        self.env.close()
        self.agent.net.load_state_dict(
        torch.load(f"best_{self.agent.name}.pth", map_location=self.agent.device)
        )
        self.record_final()
        
        x = np.arange(0, self.episodes, 1)
        plt.plot(x, reward_history)
        plt.title(f'Agent:{self.agent.name}')
        plt.savefig(f'{self.agent.name}.png')
        plt.close()

if __name__ == "__main__":
    env = gym.make("LunarLander-v3")
    agent = Reinforce(state_dim=8, action_dim=4, gamma=0.99, lr=1e-3, name='REINFORCE')
    trainer = Trainer(env, agent, episodes=3000, env_name="LunarLander-v3")
    trainer.train()