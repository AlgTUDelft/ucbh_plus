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
    import agent
    import environment  # this is required for custom environments to show up in the OpenAI Gym registry

    # Initialize the environment
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

    # Run experiments
    if verbose >= 1:
        print(f'Starting environment {env_name}.\n')
    results = []

    # Find an exact solution by solving the underlying MDP. This solution is used in plotting
    solution = pr.solve(env, discount, steps)
    if verbose >= 1:
        print(f'Value: {solution}.\n')

    # Save the results to this directory
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
