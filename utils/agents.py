from typing import Tuple, Dict, Union

import numpy as np
import torch

State = np.ndarray


class RLAgent:
    def __init__(self, env):
        self.env = env

    def on_trajectory_started(self, state: State):
        pass

    def on_trajectory_finished(self):
        pass

    def get_action(self, state: State) -> Union[int, float]:
        raise NotImplementedError

    def save_step(self, action: int, reward: float, next_state: State):
        pass

    def evaluate(self):
        pass

    def update(self) -> None:
        pass


class MemoryAgent(RLAgent):
    def __init__(self, env):
        super().__init__(env)

        self.states = []
        self.actions = []
        self.rewards = []

        self.current_ptr = 0

    @property
    def total_size(self):
        return len(self.actions)

    @property
    def current_slice(self):
        return slice(self.current_ptr, self.total_size)

    @property
    def current_indices(self):
        return range(self.current_ptr, self.total_size)

    @property
    def current_states(self):
        return self.states[self.current_slice]

    @property
    def current_actions(self):
        return self.actions[self.current_slice]

    @property
    def current_rewards(self):
        return self.rewards[self.current_slice]

    def on_trajectory_started(self, state: State):
        self.states.append(state)

    def on_trajectory_finished(self):
        self.states.pop()
        self.current_ptr = self.total_size

    def save_step(self, action: int, reward: float, next_state: State):
        self.actions.append(action)
        self.rewards.append(reward)
        self.states.append(next_state)

    def reset_memory(self):
        self.states = []
        self.actions = []
        self.rewards = []

        self.current_ptr = 0

    @property
    def tensored_data(self) -> Dict[str, torch.Tensor]:
        states = torch.as_tensor(self.states, dtype=torch.float)
        rewards = torch.as_tensor(self.rewards, dtype=torch.float)
        actions = torch.as_tensor(self.actions, dtype=torch.float)

        return {'states': states, 'rewards': rewards, 'actions': actions}

    def update(self) -> None:
        self.reset_memory()
