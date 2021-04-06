import sys
from contextlib import closing
from six import StringIO

import gym.envs.toy_text.discrete as discrete
import gym.wrappers.time_limit as tl
import gym.utils as utils
import numpy as np

from collections.abc import Iterable

DEFAULT_NUM_STATES = 40
DEFAULT_DATA = np.array([
    [2000, 1600,  50, 1.000],
    [1840, 1460,  53, 0.999],
    [1680, 1340,  56, 0.998],
    [1560, 1230,  59, 0.997],
    [1300, 1050,  62, 0.996],
    [1220,  980,  65, 0.994],
    [1150,  910,  68, 0.991],
    [1080,  840,  71, 0.988],
    [ 900,  710,  75, 0.985],
    [ 840,  650,  78, 0.983],
    [ 780,  600,  81, 0.980],
    [ 730,  550,  84, 0.975],
    [ 600,  480,  87, 0.970],
    [ 560,  430,  90, 0.965],
    [ 520,  390,  93, 0.960],
    [ 480,  360,  96, 0.955],
    [ 440,  330, 100, 0.950],
    [ 420,  310, 103, 0.945],
    [ 400,  290, 106, 0.940],
    [ 380,  270, 109, 0.935],
    [ 360,  255, 112, 0.930],
    [ 345,  240, 115, 0.925],
    [ 330,  225, 118, 0.919],
    [ 315,  210, 121, 0.910],
    [ 300,  200, 125, 0.900],
    [ 290,  190, 129, 0.890],
    [ 280,  180, 133, 0.880],
    [ 265,  170, 137, 0.865],
    [ 250,  160, 141, 0.850],
    [ 240,  150, 145, 0.820],
    [ 230,  145, 150, 0.790],
    [ 220,  140, 155, 0.760],
    [ 210,  135, 160, 0.730],
    [ 200,  130, 167, 0.660],
    [ 190,  120, 175, 0.590],
    [ 180,  115, 182, 0.510],
    [ 170,  110, 190, 0.430],
    [ 160,  105, 205, 0.300],
    [ 150,   95, 220, 0.200],
    [ 140,   87, 235, 0.100],
    [ 130,   80, 250, 0.000]
])


class ReplacementEnv(tl.TimeLimit):
    def __init__(self, nS=DEFAULT_NUM_STATES, s0=0,
                 cost=None, trade_in=None, op_cost=None, p_survival=None,
                 max_episode_steps=1000):
        super().__init__(_ReplacementEnv(nS, s0, cost, trade_in, op_cost, p_survival), max_episode_steps)
        self.P = self.env.P
        self.isd = self.env.isd
        self.nS = self.env.nS
        self.nA = self.env.nA


class _ReplacementEnv(discrete.DiscreteEnv):

    def __init__(self, nS=DEFAULT_NUM_STATES, s0=0,
                 cost=None, trade_in=None, op_cost=None, p_survival=None):

        if np.any([cost, trade_in, op_cost, p_survival] is None):
            nS = min(nS, DEFAULT_NUM_STATES)

        c = _convert_to_list(cost, nS + 1, 0)
        t = _convert_to_list(trade_in, nS + 1, 1)
        e = _convert_to_list(op_cost, nS + 1, 2)
        p = _convert_to_list(p_survival, nS + 1, 3)

        nS = min(len(c), len(t), len(e), len(p)) - 1
        nA = nS + 1

        self.s0 = 0
        isd = np.zeros(nS)
        isd[self.s0] = 1
        P = {s: {a: [] for a in range(nA)} for s in range(nS)}
        min_rew, max_rew = np.PINF, np.NINF

        for s in range(nS):

            li = P[s][0]
            prob = p[s + 1]
            next_s = min(s + 1, nS - 1)
            reward = -e[s + 1]
            if reward > max_rew:
                max_rew = reward
            if reward < min_rew:
                min_rew = reward
            if prob > 0:
                li.append((prob, next_s, reward, False))
            if prob < 1:
                li.append((1 - prob, nS - 1, reward, False))

            for a in range(1, nA):

                li = P[s][a]
                next_s = a - 1
                prob = p[a - 1]
                reward = -e[a - 1] + t[s + 1] - c[a - 1]
                if reward > max_rew:
                    max_rew = reward
                if reward < min_rew:
                    min_rew = reward
                if prob > 0:
                    li.append((prob, next_s, reward, False))
                if prob < 1:
                    li.append((1 - prob, nS - 1, reward, False))

        super().__init__(nS, nA, P, isd)

        self.reward_range = (min_rew, max_rew)

    def reset(self):

        # self.s0 = (self.s0 + 1) % self.nS
        # isd = np.zeros(self.nS)
        # isd[self.s0] = 1
        return super().reset()

    def render(self, mode='human', policy=None, **kwargs):

        outfile = StringIO() if mode == 'ansi' else sys.stdout
        nS = self.nS

        def s_str(state):
            x_val = (state + 1) / nS
            if x_val == 0:
                res = 'new'
            elif x_val == 1:
                res = 'broken'
            else:
                res = '{:2.0f}% used'.format(x_val * 100)
            return res

        def a_str(action):
            if action == 0:
                res = 'do not replace'
            else:
                res = 'replace with a {} car'.format(s_str(action - 1))
            return res

        if policy is None:
            result = 'Car state: {}. Last action: {}.'.format(s_str(self.s), a_str(self.lastaction))
        else:
            lines = ['---']
            for s in range(nS):
                string = '{} → {}'.format(s_str(s).ljust(8), a_str(policy[s]))
                if s != self.s:
                    string = utils.colorize(string, 'white')
                lines.append(string)
            lines.append('---')
            result = '\n'.join(lines)

        outfile.write(result + '\n')

        if mode != 'human':
            with closing(outfile):
                return outfile.getvalue()


def _convert_to_list(x, length, i):

    n_start = 0
    n_end = length + n_start

    if x is None:
        by = 40 // (length - 1)
        dat = np.reshape(DEFAULT_DATA[1:41, i], (40 // by, by))
        dat = dat[:, by-1]
        result = np.append(np.array(DEFAULT_DATA[0,i]), dat)
    elif np.isscalar(x):
        result = np.full(length, x)
    elif isinstance(x, Iterable):
        result = list(x)[n_start:n_end]
    elif callable(x):
        result = [x(i / (length - 1.0)) for i in range(n_start, n_end)]
    else:
        raise ValueError

    return result
