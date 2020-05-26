import numpy as np

State = np.ndarray


class RLAgent:
    def __init__(self, env):
        pass

    def on_trajectory_started(self, state: State) -> None:
        pass

    def on_trajectory_finished(self) -> None:
        pass

    def get_action(self, state: State) -> int:
        raise NotImplementedError

    def save_step(self, reward: float, next_state: State) -> None:
        pass

    def update(self) -> None:
        raise NotImplementedError
