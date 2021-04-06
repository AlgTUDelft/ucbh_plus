import gym.envs.toy_text.frozen_lake as fl
import gym.wrappers.time_limit as tl


class FrozenLakeAdjustableEnv(tl.TimeLimit):
    def __init__(self, desc=None, map_name='8x8', p_follow=1.0, rewards=None, terminate_in_holes=True,
                 max_episode_steps=1000):
        super().__init__(_FrozenLakeAdjustableEnv(
            desc, map_name, p_follow, rewards, terminate_in_holes), max_episode_steps
        )
        self.P = self.env.P
        self.isd = self.env.isd
        self.nS = self.env.nS
        self.nA = self.env.nA
        self._max_episode_steps = max_episode_steps


class _FrozenLakeAdjustableEnv(fl.FrozenLakeEnv):
    """
    This is an implementation of slippery grid world with holes (aka frozen lake),
    but with adjustable rewards and transition probabilities.
    """
    def __init__(self, desc=None, map_name='8x8', p_follow=1.0, rewards=None, terminate_in_holes=True):

        rew = {'hole': 0.0, 'goal': 1.0, 'step': 0.0} if rewards is None else rewards

        # Initialize default Frozen Lake
        is_slippery = p_follow < 1.0
        super().__init__(desc, map_name, is_slippery)
        self.reward_range = (min(rew.values()), max(rew.values()))

        # Adjust rewards and transition probabilities
        p_drift = (1.0 - p_follow) / 2
        for i in self.P:
            letter = self._index_to_letter(i)
            for a in range(4):
                action = self.P[i][a]
                for j in range(len(action)):
                    # Find new rewards
                    if letter == b'H':
                        new_reward = rew['hole']
                        new_done = terminate_in_holes
                    elif letter == b'G':
                        new_reward = rew['goal']
                        new_done = True
                    else:
                        new_reward = rew['step']
                        new_done = False

                    # Find new transition probabilities
                    if is_slippery and len(action) != 1:
                        new_p = p_follow if j == 1 else p_drift
                    else:
                        new_p = 1.0

                    # Adjust the problem data accordingly
                    action[j] = (new_p, action[j][1], new_reward, new_done)

    def _index_to_letter(self, index):
        row = index // self.ncol
        col = index % self.ncol
        return self.desc[row, col]
