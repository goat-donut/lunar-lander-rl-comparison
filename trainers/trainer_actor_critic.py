from agents.actor_critic import ActorCritic
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

    def record_final(self):
        env = gym.make(self.env_name, render_mode='rgb_array')
        state, _ = env.reset()
        frames = []

        while True:
            frame = env.render()
            frames.append(frame)
            action, _, _ = self.agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action.item())
            done = terminated or truncated
            state = next_state
            if done:
                break

        env.close()
        imageio.mimsave(f"result_{self.agent.name}.gif", frames, fps=30)

    def train(self):
        reward_history = []
        best_reward = -np.inf
        N_STEPS = 10

        for ep in range(self.episodes):
            state, _ = self.env.reset()
            total_reward = 0
            
            states, log_probs, rewards, next_states, dones, dists = [], [], [], [], [], []

            while True:
                action, log_prob, dist = self.agent.select_action(state)
                next_state, reward, terminated, truncated, _ = self.env.step(action.item())
                done = terminated or truncated

                states.append(state)
                log_probs.append(log_prob)
                rewards.append(reward)
                next_states.append(next_state)
                dones.append(float(done))
                dists.append(dist)

                state = next_state
                total_reward += reward

                if len(states) == N_STEPS or done:
                    self.agent.train(log_probs, rewards, states, next_states, dones, dists)
                    states, log_probs, rewards, next_states, dones, dists = [], [], [], [], [], []

                if done:
                    break

            reward_history.append(total_reward)
            if (ep + 1) >= 200:
                avg_reward = np.mean(reward_history[-200:])
                if avg_reward > best_reward:
                    best_reward = avg_reward
                    torch.save(self.agent.actor.state_dict(), f"best_{self.agent.name}_actor.pth")
                    torch.save(self.agent.critic.state_dict(), f"best_{self.agent.name}_critic.pth")

            if (ep + 1) % 200 == 0:
                avg_reward = np.mean(reward_history[-200:])
                print(f"Episode {ep+1} | Avg reward: {avg_reward:.3f} | Best: {best_reward:.3f}")

        self.env.close()
        self.agent.actor.load_state_dict(
            torch.load(f"best_{self.agent.name}_actor.pth", map_location=self.agent.device)
        )
        self.record_final()

        x = np.arange(self.episodes)
        plt.plot(x, reward_history)
        plt.title(f'Agent: {self.agent.name}')
        plt.savefig(f'{self.agent.name}.png')
        plt.close()


if __name__ == "__main__":
    env = gym.make("LunarLander-v3")
    agent = ActorCritic(
        state_dim=8,
        action_dim=4,
        gamma=0.99,
        lr_actor=3e-4,
        lr_critic=1e-3,
        name='ActorCritic'
    )
    trainer = Trainer(env, agent, episodes=5000, env_name="LunarLander-v3")
    trainer.train()