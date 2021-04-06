from abc import ABC
from gym.core import Env
from gym.envs.toy_text.discrete import DiscreteEnv
from progressbar import progressbar
from .policy import Policy
import numpy as np
import scipy.stats as st


class Agent:

    def __init__(self, env: Env, policy: Policy = None, name=None, verb=0):

        self._env = env
        if self._env is not None:
            self._env.reset()

        self._policy = policy
        if self._policy is not None:
            self._policy.reset()

        self._verboseness = verb
        self.name = name

    def reset_environment(self):
        if self._env is not None:
            self._env.reset()
        if self._policy is not None:
            self._policy.reset()

    def set_verboseness(self, verboseness):
        self._verboseness = verboseness

    def run(self, num_episodes):

        if self._verboseness >= 1:
            self._talk(message=f'Agent {self.name} started learning.')

        episode_range = range(num_episodes)
        if self._verboseness >= 1:
            episode_range = progressbar(episode_range)

        for episode in episode_range:

            # initialize episode
            done = False
            observation = self._initialize_episode()

            while not done:

                # chose an action
                action = self._get_action(observation)

                # perform an action, get data from the environment
                next_observation, reward, done, info = self._step(observation, action)

                # learn from the observation
                self._learn(observation, action, next_observation, reward, done, info)

                # proceed to the next step
                observation = next_observation

                # finish the step
                self._wrap_up_step()

            # finish the episode
            self._wrap_up_episode(episode)

        # finish the run
        self._wrap_up_run()
        return self._run_results()

    def learned_policy(self):
        pass

    def get_stats(self, episode=None):
        return {}

    # this are the methods that need to be implemented for different agents
    def _initialize_episode(self):
        s = self._env.reset()
        return s

    def _talk(self, message=None, render=False):
        if message is not None:
            print(message)
        if render:
            self._env.render()

    def _get_action(self, observation):
        return self._policy.get_action(observation)

    def _learn(self, observation, action, next_observation, reward, done, info, **kwargs):
        pass

    def _step(self, observation, action):
        return self._env.step(action)

    def _wrap_up_step(self):
        pass

    def _wrap_up_episode(self, episode):
        # if verbose, output data to console
        if self._verboseness >= 3:
            message = f'End of episode #{episode}.'
            self._talk(message=message, render=self._verboseness >= 4)
            message = 'Episode stats:\n'
            stats = self.get_stats(episode=episode)
            m = []
            for k in stats:
                stat = stats[k]
                m.append(f'{k}: {stat:4.2f}' if type(stats[k]) is float else f'{k}: {stat}')
            message += ', '.join(m)
            message += '\n'
            self._talk(message=message)
        self._policy.update()

    def _wrap_up_run(self):
        if self._verboseness >= 1:
            message = f'Agent {self.name} finished learning.'
            self._talk(message=message, render=self._verboseness >= 2)
            message = 'Trial stats (mean ± standard error):\n'
            stats = self.get_stats()
            m = []
            stat_names = stats[0].keys()
            n = len(stats)
            for k in stat_names:
                if k == 'episode':
                    continue
                stat = np.average([stats[i][k] for i in range(n)])
                sem = st.sem([stats[i][k] for i in range(n)])
                m.append(f'{k}: {stat:4.2f} ± {sem:4.2f}')
            message += ', '.join(m)
            message += '\n'
            self._talk(message=message)

    def _run_results(self):
        return None


class DiscreteAgent(Agent, ABC):

    def __init__(self, env: DiscreteEnv, policy: Policy = None, name=None, verb=0):
        super().__init__(env, policy, name, verb)
        if env is not None:
            self._nS = env.nS
            self._nA = env.nA
