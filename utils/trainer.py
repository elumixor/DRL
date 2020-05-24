# For each epoch, we will play several rollouts, collect samples and train
import time
from collections import namedtuple
from typing import Tuple, Optional, Callable, List, NamedTuple

from numpy import mean, sum
import torch

from utils import Plotter

Rollout = NamedTuple('Rollout',
                     [('states', torch.Tensor),
                      ('actions', torch.Tensor),
                      ('rewards', torch.Tensor),
                      ('next_states', torch.Tensor)])


def train(env, get_action: Callable[[List[float]], int], update_agent: Callable[[List[Rollout]], None],
          epochs: int = 100, num_trajectories: int = 1, print_frequency: int = 1,
          render_frequency: Optional[int] = None, plot_frequency: Optional[int] = None, max_timesteps: int = -1):
    """
    Generalized training function.

    --------------------------------------------------------------------------------------------------------------------

    For each epoch, creates several rollouts (trajectories) of the given environment.

    Requires get_action(state) function to determine an action for the given state and let an agent act in the
    environment, and update_agent(trajectories), that is called at the end of an epoch - when all rollouts have finished
    to train an agent.

    --------------------------------------------------------------------------------------------------------------------

    :param env: OpenAI gym environment

    :param get_action: Decision function for the agent, accepts a state and returns an action an agent will take

    :param update_agent: Function that updates an agent at the end of an epoch. Accepts a list of trajectories

    :param epochs: Number of epochs to train

    :param num_trajectories: Number of trajectories per epoch

    :param print_frequency: How often will the results be printed.
                            For example, if set to 5, will print results every 5 epochs

    :param render_frequency: How often will the environment be rendered. This number is shared across trajectories
                             and epochs. For example, if set to 5, and number of trajectories is 3, will render at
                             (epoch 0, traj. 0), (epoch 1, traj. 1), (epoch 2, traj. 0)  ...

    :param max_timesteps: Maximum timesteps per trajectory. If the rollout is too long (for example, when agent performs
                          well, this will cut of the rollout, so we don't have infinitely long rollouts

    :param plot_frequency: How often should the results be plotted

    """
    # assert
    if max_timesteps < 0:
        max_timesteps = float('inf')
    plotter = Plotter()
    plotter['reward'].name = "Mean total epoch reward"

    total_time = 0.
    epochs_time = 0.

    start_time = time.time()

    global_rollout = 0
    for epoch in range(epochs):
        rollouts = []
        rollout_rewards = []

        for rollout in range(num_trajectories):
            state = env.reset()
            done = False

            samples = []
            total_reward = 0

            t = 0
            while not done and t < max_timesteps:
                if render_frequency and global_rollout % render_frequency == 0:
                    env.render()

                with torch.no_grad():
                    action = get_action(state)

                next_state, reward, done, _ = env.step(action)

                samples.append((state, action, reward, next_state))
                total_reward += reward

                state = next_state

                t += 1

            # Transpose our samples
            states, actions, rewards, next_states = zip(*samples)

            states = torch.stack([torch.from_numpy(state) for state in states], dim=0).float()
            next_states = torch.stack([torch.from_numpy(state) for state in next_states], dim=0).float()
            actions = torch.as_tensor(actions).unsqueeze(1)
            rewards = torch.as_tensor(rewards).unsqueeze(1)

            rollouts.append(Rollout(states, actions, rewards, next_states))

            rollout_rewards.append(total_reward)

            global_rollout += 1

        update_agent(rollouts)

        plotter['reward'] += mean(rollout_rewards)

        end_time = time.time()
        epochs_time += end_time - start_time
        start_time = end_time

        if epoch % print_frequency == 0:
            total_time += epochs_time

            print(f'Epoch\t{epoch} \t| '
                  f'{total_time:.02f}s \t| '
                  f'Mean total reward\t{mean(plotter["reward"].y[-print_frequency:]):.4f}')

            epochs_time = 0.

        if plot_frequency is not None and epoch % plot_frequency == 0:
            plotter.show("reward", running_average=True)
