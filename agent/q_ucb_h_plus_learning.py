from . import QUCBHLearningAgent
import numpy as np
import math
import gym.wrappers.time_limit as tl


class QUCBHPlusLearningAgent(QUCBHLearningAgent):

    def __init__(self,
                 env: tl.TimeLimit, name=None, verb=0, discount=None,
                 detect_terminals=True, learning_rate=None,
                 delta=0.001, c=0.001, num_episodes=10000, lam=1.0, omega=0.8,
                 ):
        super().__init__(env, name, verb, discount, detect_terminals, learning_rate, delta, c, num_episodes)
        self._lambdaH = lam * self._H
        self._omega = omega

    def _learn(self, observation, action, next_observation, reward, done, info, **kwargs):
        # if agent knows how to detect terminals, use zero Q-value for the next state value
        step = self.current_step()
        if done and self._detect_terminals:
            next_q = 0.0
            self._q[step + 1][next_observation] = np.zeros((self._nA,))
        else:
            next_q = float(min(max(self._q[step+1][next_observation]), self._starting_q))

        # update the Q-table
        t = self._n_visits[step][observation][action]
        v_next = self._reward_range * (self._H - step + 1 if self._discount == 1 else
                                       (1 - self._discount ** (self._H - step + 1))/(1 - self._discount))
        alpha = self._s_alpha(t)
        bonus = 1.0 / alpha * self._bonus_base(t) + (1.0 - 1.0 / alpha) * self._bonus_base(t - 1)
        bonus *= self._c * v_next * self._discount * math.sqrt(self._iota)
        update = reward + self._discount * next_q + bonus - self._q[step][observation][action]
        self._q[step][observation][action] += alpha * update

    def _s_alpha(self, t):
        return (self._lambdaH + 1.0) / (self._lambdaH + t ** self._omega)

    def _bonus_base(self, t):
        return 1.0 / math.sqrt((self._lambdaH + t) ** self._omega)

    def learned_policy(self):
        step = self.current_step()
        return [np.argmax(self._q[step][state]) for state in range(self._nS)]
