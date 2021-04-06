from .agent import DiscreteAgent
import gym.wrappers.time_limit as tl
import numpy as np
from itertools import accumulate

DEFAULT_DISCOUNT = 1.0
DEFAULT_DETECT = True


def _default_alpha(h):
    return lambda n_visits: (h + 1.0) / (h + n_visits)


class EpisodicQLearningAgent(DiscreteAgent):

    def __init__(self, env: tl.TimeLimit, policy=None, name=None, verb=0, discount=None,
                 starting_q=0.0, detect_terminals=True, learning_rate=None):

        super().__init__(env.unwrapped, policy, name, verb)
        self._env, self._unwrapped_env = env, env.unwrapped
        self._discount = DEFAULT_DISCOUNT if discount is None else discount
        assert float('-inf') < starting_q < float('inf'), 'Starting Q-value for Q-learning must be finite'
        self._starting_q = starting_q
        self._nH = env._max_episode_steps

        self._detect_terminals = detect_terminals
        self._alpha = _default_alpha(self._nH) if learning_rate is None else learning_rate
        self._episode_rewards = [[]]

        if self._env is not None:
            self._fill_q()

        self._n_visits = np.zeros((self._nH, self._nS, self._nA), dtype=int)

    def reset_environment(self):
        super().reset_environment()
        self._episode_rewards = [[]]
        self._fill_q()
        self._n_visits = np.zeros((self._nH, self._nS, self._nA), dtype=int)

    def _step(self, observation, action):
        h = self._env._elapsed_steps
        self._n_visits[h][observation][action] += 1

        next_state, reward, done, info = super()._step(observation, action)

        # accumulate rewards
        if self._episode_rewards[-1] is None:
            self._episode_rewards[-1] = []
        self._episode_rewards[-1].append(reward)
        return next_state, reward, done, info

    def _get_action(self, observation):
        action = self._policy.get_action((self._env._elapsed_steps, observation), self._q)
        return action

    def _wrap_up_episode(self, episode):
        super()._wrap_up_episode(episode)
        self._episode_rewards.append([])

    def _run_results(self):
        return self._episode_rewards

    def get_stats(self, episode=None):
        if episode is not None:
            stats = {
                'total reward': np.sum(self._episode_rewards[episode]),
                'discounted total reward': _discounted_sum(self._episode_rewards[episode], self._discount),
                'length': len(self._episode_rewards[episode])
            }
        else:
            rews = self._episode_rewards[:-1]
            stats = [{
                'episode': i,
                'total reward': np.sum(rews[i]),
                'discounted total reward': _discounted_sum(rews[i], self._discount),
                'episode length': len(rews[i])
            } for i in range(len(rews))]
        return stats

    def _fill_q(self):
        if np.isscalar(self._starting_q):
            self._q = np.full((self._nH + 1, self._nS, self._nA), float(self._starting_q))
        else:
            self._q = np.tile(self._starting_q, (self._nH + 1, self._nS, self._nA, 1))
        self._q[-1] = np.zeros((self._nS, self._nA))

    def current_step(self):
        return self._env._elapsed_steps - 1


def _discounted_sum(rewards, discount):
    if len(rewards) == 0:
        return 0
    reversed_rewards = rewards[::-1]
    acc = list(accumulate(reversed_rewards, lambda x, y: float(x) * discount + y))
    return acc[-1]
