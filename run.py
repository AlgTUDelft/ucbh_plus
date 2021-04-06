# -*- coding: utf-8 -*-
from typing import Union, Optional
import process_results as pr
import os
from datetime import datetime
from gym import Env, make
from gym.wrappers import TimeLimit


def run(
        env: Union[Env, str],
        trials: int = 1,
        episodes: int = 10000,
        steps: Optional[int] = None,
        discount: float = 1.0,
        starting_q: Optional[float] = None,
        exploration_rate: float = 1.0,
        exploration_rate_decay: float = 0.99,
        min_exploration_rate: float = 0.00,
        delta: float = 0.001,
        c: float = 0.0,
        lamb: float = 1.0,
        omega: float = 0.8,
        verbose: Optional[int] = None,
        save: bool = True,
        save_dir: str = 'results',
        plot: bool = True,
        smoothing: float = 0.05,
        iqr: float = 0.0
):
    """
    Runs three agents (UCB-H+, UCB, and Q-Learning) in a given environment
    :param env: Environment: either an OpenAI Gym environment or a string;
    in the latter case gym.make(env) will be used.
    :param trials: Number of trials to run
    :param episodes: Number of episodes in each trial
    :param steps: Number of time steps per episode
    :param discount: Discounting factor
    :param starting_q: Initial Q-values for Q-Learning. UCB-H+ and UCB-H infer these from the environment
    :param exploration_rate: Initial exploration rate for Q-Learning
    :param exploration_rate_decay: Exploration rate decay for Q-Learning; each time step the exploration rate is
    multiplied by this until it reaches the minimum exploration rate
    :param min_exploration_rate: Minimum exploration rate for Q-Learning
    :param delta: PAC-probability delta for UCB-H and UCB-H+
    :param c: UCB-constant c for UCB-H and UCB-H+
    :param lamb: lambda coefficient for UCB-H+; this is added to the numerator and demoninator of the learning rate
    :param omega: power coefficient for UCB-H+
    :param verbose: how much information to output to console: from 0 (None) to 4 (a lot of information)
    :param save: whether to save the experiment data or not
    :param save_dir: directory where the data will be saved
    :param plot: whether to plot the experiment data or not
    :param smoothing: moving-average smoothing relative to the number of episodes
    :param iqr: inter-quantile range for plotting
    """
    import agent
    import environment  # this is required for custom environments to show up in the OpenAI Gym registry

    # Initialize the environment an make it a TimeLimit environment for episodic learning
    if isinstance(env, str):
        env_name = env
        env = make(env_name)
    else:
        env_name = type(env).__name__
    if not isinstance(env, TimeLimit):
        assert steps is not None, 'The number of steps per episode is not given'
        env = TimeLimit(env, steps)
    else:
        if steps is None:
            steps = env._max_episode_steps
        else:
            env._max_episode_steps = steps

    # If verbosity is not given, use 0, i.e., no console output
    if verbose is None:
        verbose = 0

    # If starting Q-value for Q-learning is not supplied, try to infer it from the environment
    if starting_q is None:
        reward_max = float(env.reward_range[1])
        starting_q = reward_max / (1.0 - discount) if discount < 1 else reward_max * steps

    # Initialize the agents
    agents = [
        agent.QUCBHPlusLearningAgent(
            env=env,
            name='QUCBPlus',
            verb=verbose,
            discount=discount,
            delta=delta,
            c=c,
            lam=lamb,
            omega=omega
        ),
        agent.QUCBHLearningAgent(
            env=env,
            name='QUCB',
            verb=verbose,
            discount=discount,
            delta=delta,
            c=c,
        ),
        agent.SimpleQLearningAgent(
            env=env,
            policy=agent.policy.EpsilonGreedyPolicy(exploration_rate, exploration_rate_decay, min_exploration_rate),
            name='Q_max',
            verb=verbose,
            discount=discount,
            starting_q=starting_q
        )
    ]

    # Start the experiments
    if verbose >= 1:
        print(f'Starting environment {env_name}.\n')
    results = []

    # Find an exact solution by solving the underlying MDP. This solution is used in plotting
    solution = pr.solve(env, discount, steps)
    if verbose >= 1:
        print(f'Value: {solution}.\n')

    # Build a path to save directory
    path = os.path.join(save_dir, env_name, datetime.now().strftime('%Y-%m-%d-%H-%M')) if save else None

    # Run the trials
    for trial in range(trials):

        if verbose >= 1:
            print(f'Starting trial #{trial}.\n')

        # Run each agent for the given number of episodes
        for agent in agents:
            agent.reset_environment()
            agent.run(episodes)

            result = agent.get_stats()
            result = [{**{'method': agent.name, 'trial': trial}, **r} for r in result]

            if save:
                pr.save(path, env_name, result)

            results.extend(result)

        if verbose >= 1:
            print(f'Trial #{trial} done.\n')

    # Add the solution to the results file
    if save:
        solution_dict = {key: '' for key in results[0].keys()}
        solution_dict['method'] = 'Solution'
        solution_dict['trial'] = 0
        solution_dict['episode'] = 0
        solution_dict['discounted total reward'] = solution
        pr.save(path, env_name, [solution_dict])

    # Visualize the data if plotting is on
    if plot:
        results = pr.fetch_stat(results, 'total reward', episodes, trials)
        plot_quantiles = iqr is not None and 0.0 < iqr <= 1.0
        pr.plot(results, title='{}, d={}'.format(env_name, discount), solution=solution, episodes=episodes,
                ma=int(smoothing * episodes), show_q=plot_quantiles, iqr=iqr)
