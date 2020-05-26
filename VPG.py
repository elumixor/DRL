import gym
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical
from torch.optim import Adam

from utils import train, torch_device, rewards_to_go, normalize, RLAgent


class Agent(RLAgent):
    def __init__(self, env):
        super().__init__(env)

        obs_size = env.observation_space.shape[0]
        num_actions = env.action_space.n

        # Hyper parameters
        hidden_actor = 16
        self.discounting = 0.99

        # Actor maps state to actions' probabilities
        self.actor = nn.Sequential(nn.Linear(obs_size, hidden_actor),
                                   nn.ReLU(),
                                   nn.Linear(hidden_actor, num_actions),
                                   nn.Softmax(dim=1))

        # Optimizers
        self.optimizer = Adam(self.actor.parameters(), lr=0.01)

        self.rollouts = []
        self.rollout_samples = []

    def on_trajectory_finished(self) -> None:
        states, actions, rewards, next_states = zip(*self.rollout_samples)

        states = torch.stack([torch.from_numpy(s) for s in states]).float().to(torch_device)
        next_states = torch.stack([torch.from_numpy(s) for s in next_states]).float().to(torch_device)
        actions = torch.as_tensor(actions, device=torch_device).unsqueeze(1)
        rewards = torch.as_tensor(rewards, dtype=torch.float, device=torch_device).unsqueeze(1)

        self.rollouts.append((states, actions, rewards, next_states))
        self.rollout_samples = []

    def save_step(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray) -> None:
        self.rollout_samples.append((state, action, reward, next_state))

    def get_action(self, state: np.ndarray) -> int:
        state = torch.from_numpy(state).float().unsqueeze(0)
        probabilities = self.actor(state)
        action = Categorical(probabilities).sample()
        return action.item()

    def update(self) -> None:
        self.actor.train().to(torch_device)

        loss = 0
        total_len = 0

        for states, actions, rewards, _ in self.rollouts:
            # Simplest strategy: use the total reward
            # weights = sum(rewards)

            # Improvement: use discounted rewards to go
            weights = rewards_to_go(rewards, self.discounting).flatten()
            weights = normalize(weights)

            # Get probabilities, shape (episode_length * num_actions)
            # Then select only the probabilities corresponding to sampled actions
            probabilities = self.actor(states)
            probabilities = probabilities[range(states.shape[0]), actions.flatten()]
            loss += (-probabilities.log() * weights).sum()

            # Take the weighted average (helps convergence)
            total_len += weights.shape[0]

        self.step(loss / total_len)
        self.actor.eval().to('cpu')

        self.rollouts = []

    def step(self, loss: torch.Tensor) -> None:
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


# Train, provide an env, function to get an action from state, and training function that accepts rollouts
train(gym.make('CartPole-v0'), Agent,
      epochs=2000, num_trajectories=5, print_frequency=10, plot_frequency=50, render_frequency=500)
