from .frozen_lake import FrozenLakeAdjustableEnv
from .replacement import ReplacementEnv
from gym.envs.registration import register


__all__ = ['FrozenLakeAdjustableEnv', 'ReplacementEnv']

register(
    id='Replacement-v0',
    entry_point='environment.replacement:ReplacementEnv',
    max_episode_steps=40,
    reward_threshold=10,
    nondeterministic=False,
)

register(
    id='Lake-v0',
    entry_point='environment.frozen_lake:FrozenLakeAdjustableEnv',
    reward_threshold=1.0,
    nondeterministic=False,
    kwargs={'map_name': '8x8', 'p_follow': 1.0, 'max_episode_steps': 16}
)
