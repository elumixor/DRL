# For each epoch, we will play several rollouts, collect samples and train
import time

import numpy as np
import torch

from utils import Plotter


def train(env, get_action, train_epoch, epochs=100, num_rollouts=10, print_epochs=10, render=False):
    if isinstance(render, bool):
        render_part = False
    else:
        render_part = True
        render_epochs, render_times = render

    plotter = Plotter()
    plotter['reward'].name = "Mean total epoch reward"

    total_time = 0.
    epochs_time = 0.

    start_time = time.time()

    for epoch in range(epochs):
        with torch.no_grad():
            rollouts = []
            rewards = []

            for rollout in range(num_rollouts):
                state = env.reset()
                done = False

                samples = []
                total_reward = 0

                while not done:
                    if not render_part and render or render_part and epoch % render_epochs == 0 and rollout < render_times:
                        env.render()

                    action = get_action(state)
                    next_state, reward, done, _ = env.step(action)

                    samples.append((state, action, reward, next_state))
                    total_reward += reward

                    state = next_state

                rollouts.append(samples)
                rewards.append(total_reward)

        train_epoch(rollouts)

        plotter['reward'] += np.mean(rewards)

        end_time = time.time()
        epochs_time += end_time - start_time
        start_time = end_time

        if epoch % print_epochs == 0:
            total_time += epochs_time

            print(f'Epoch\t{epoch}\t | '
                  f'{total_time:.02f}s | \t'
                  f'Mean total reward\t{np.mean(plotter["reward"].y[-print_epochs:])}')

            epochs_time = 0.
            plotter.show("reward", running_average=True)
