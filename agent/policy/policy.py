import numpy as np
import random
from abc import abstractmethod


class Policy:
    def __init__(self):
        pass

    @abstractmethod
    def get_action(self, observation, q=None):
        return NotImplemented

    @abstractmethod
    def update(self):
        return NotImplemented

    @abstractmethod
    def reset(self):
        return NotImplemented


class EpsilonGreedyPolicy(Policy):
    def __init__(self, starting_epsilon: float = 1.0, epsilon_decay: float = 0.999, min_epsilon: float = 0.01):
        """
        Epsilon-greedy policy with decaying learning rate
        :param starting_epsilon: Starting learning rate
        :param epsilon_decay: Every step the learning rate is multiplied by this amount until it reaches the minimum
        :param min_epsilon: Minimum learning rate
        """
        super(EpsilonGreedyPolicy, self).__init__()
        self._eps0 = self._eps = starting_epsilon
        self._decay = epsilon_decay
        self._min = min_epsilon

    def get_action(self, observation, q=None):

        if random.uniform(0, 1) < self._eps:
            # explore randomly
            action = np.random.randint(0, len(q[observation]))
        else:
            # exploit learned values
            q_obs = q[observation]
            action = np.random.choice(np.flatnonzero(q_obs == q_obs.max()))
        return action

    def update(self):
        self._eps = max(self._min, self._eps * self._decay)

    def reset(self):
        self._eps = self._eps0


class UCBPolicy(Policy):

    def get_action(self, observation, q=None):
        qq = q[observation]
        action = np.argmax(qq)
        return action

    def update(self):
        pass

    def reset(self):
        pass
