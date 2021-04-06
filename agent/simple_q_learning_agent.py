from .episodic_q_learning_agent import EpisodicQLearningAgent
import numpy as np


class SimpleQLearningAgent(EpisodicQLearningAgent):

    def _learn(self, observation, action, next_observation, reward, done, info):
        # if agent knows how to detect terminals, use zero Q-value for the next state value
        step = self._env._elapsed_steps - 1
        if done and self._detect_terminals:
            next_q = 0.0
        else:
            next_q = max(self._q[step+1][next_observation])

        # update the Q-table
        update = reward + self._discount * next_q - self._q[step][observation][action]
        alpha = self._alpha(self._n_visits[step][observation][action])
        self._q[step][observation][action] += alpha * update

    def learned_policy(self):
        step = self._env._elapsed_steps - 1
        return [np.argmax(self._q[step][state]) for state in range(self._nS)]
