import gymnasium as gym
from gymnasium import spaces
import numpy as np


class TwoAZeroObsOneStep(gym.Env):
    def __init__(self):
        super().__init__()

        self.observation_space = spaces.Discrete(1)
        self.action_space = spaces.Discrete(2)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        obs = 0
        info = {}

        return obs, info

    def step(self, action):
        reward = 1.0 if action == 0 else -1.0

        terminated = True
        truncated = False

        obs = 0
        info = {}

        return obs, reward, terminated, truncated, info


class TwoARandomObsOneStep(gym.Env):
    def __init__(self):
        super().__init__()

        self.observation_space = spaces.Discrete(2)
        self.action_space = spaces.Discrete(2)

        self.state = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.state = self.np_random.integers(0, 2)

        return int(self.state), {}

    def step(self, action):
        # En s=0 la acción correcta es 0.
        # En s=1 la acción correcta es 1.
        reward = 1.0 if action == self.state else -1.0

        terminated = True
        truncated = False

        obs = int(self.state)

        return obs, reward, terminated, truncated, {}


class LineWorldEasyEnv(gym.Env):
    def __init__(self):
        super().__init__()

        # Internamente usamos 0..5.
        # Conceptualmente corresponden a las posiciones 1..6.
        self.observation_space = spaces.Discrete(6)
        self.action_space = spaces.Discrete(2)

        self.state = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.state = 0

        return self.state, {}

    def step(self, action):
        if action == 0:  # izquierda
            self.state = max(0, self.state - 1)

        elif action == 1:  # derecha
            self.state = min(5, self.state + 1)

        terminated = self.state == 5
        truncated = False

        reward = 1.0 if terminated else 0.0

        return self.state, reward, terminated, truncated, {}


class LineWorldMirrorEnv(gym.Env):
    def __init__(self):
        super().__init__()

        # Estados conceptuales 1..4
        # Internamente 0..3
        self.observation_space = spaces.Discrete(4)
        self.action_space = spaces.Discrete(2)

        self.state = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.state = 0

        return self.state, {}

    def step(self, action):
        # Estado conceptual 2 -> índice 1.
        # En este estado las acciones están invertidas.
        if self.state == 1:
            if action == 0:      # "izquierda" mueve a la derecha
                self.state += 1
            elif action == 1:    # "derecha" mueve a la izquierda
                self.state -= 1

        else:
            if action == 0:      # izquierda normal
                self.state = max(0, self.state - 1)
            elif action == 1:    # derecha normal
                self.state = min(3, self.state + 1)

        reward = -1.0

        terminated = self.state == 3
        truncated = False

        return self.state, reward, terminated, truncated, {}