from .episodic_q_learning_agent import EpisodicQLearningAgent
import numpy as np
import math
from .policy import UCBPolicy
import gym.wrappers.time_limit as tl


class QUCBHLearningAgent(EpisodicQLearningAgent):

    def __init__(self,
                 env: tl.TimeLimit, name=None, verb=0, discount=None, detect_terminals=True,
                 delta=0.001, c=0.001, num_episodes=10000):
        """
        UCB-H agent
        :param env: OpenAI Gym environment; must be gym.wrappers.time_limit.TimeLimit for episodic learning
        :param name: Agent's name for displaying
        :param verb: Verbosity
        :param discount: Discounting factor
        :param detect_terminals: Can the agent detect that the episode is terminated from the 'done' signal?
        :param delta: PAC-probability delta
        :param c: UCB-constant c
        :param num_episodes: number of episodes to run
        """
        policy = UCBPolicy()
        H = env._max_episode_steps
        assert env.reward_range[1] < float('inf') and env.reward_range[0] > float('-inf'),\
            'environment must have a finite reward range for UCB-learning to work.'
        starting_q = env.reward_range[1] * H
        super().__init__(env, policy, name, verb, discount, starting_q, detect_terminals)
        self._H = H
        self._c = c
        self._delta = delta
        self._K = num_episodes
        self._iota = math.log(self._nS * self._nA * self._H * self._K / self._delta)
        self._reward_range = env.reward_range[1] - env.reward_range[0]

    def _learn(self, observation, action, next_observation, reward, done, info):
        # if agent knows how to detect terminals, use zero Q-value for the next state value
        step = self.current_step()
        if done and self._detect_terminals:
            next_q = 0.0
            self._q[step + 1][next_observation] = np.zeros((self._nA,))
        else:
            next_q = float(min(max(self._q[step+1][next_observation]), self._starting_q))

        # update the Q-table
        t = self._n_visits[step][observation][action]
        bonus = self._c * self._reward_range * math.sqrt(8 * self._H * self._iota / t)
        update = reward + self._discount * next_q + bonus - self._q[step][observation][action]
        alpha = self._alpha(t)
        self._q[step][observation][action] += alpha * update

    def learned_policy(self):
        step = self.current_step()
        return [np.argmax(self._q[step][state]) for state in range(self._nS)]
