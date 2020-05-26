import numpy as np


class RLAgent:
    def __init__(self, env):
        pass

    def on_trajectory_started(self) -> None:
        pass

    def on_trajectory_finished(self) -> None:
        pass

    def get_action(self, state: np.ndarray) -> int:
        raise NotImplementedError

    def save_step(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray) -> None:
        pass

    def update(self) -> None:
        raise NotImplementedError
