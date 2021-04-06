import argparse
from run import run
import yaml


def parse_args():
    """
    parse the arguments
    :return: (dict) the arguments
    """
    parser = argparse.ArgumentParser(description='Run UCB-H, UCB-H+, and Q-Learning agents in a given environment')

    parser.add_argument('env', help='Environment. Must be registered in the OpenAI Gym', type=str)

    parser.add_argument('--trials', help='Number of trials', type=int)
    parser.add_argument('--episodes', help='Number of episodes (K)', type=int)
    parser.add_argument('--steps', help='Number of steps (H)', type=int)

    parser.add_argument('--discount', help='Discounting rate (gamma)', type=float)
    parser.add_argument('--starting_q', help='Starting Q-values for Q-learning. UCB-H and UCB-H+ infer these'
                                             'from the environment automatically', type=float)

    parser.add_argument('--exploration_rate', help='Starting exploration rate for Q-learning',
                        type=float)
    parser.add_argument('--exploration_rate_decay', help='Exploration rate decay for Q-learning',
                        type=float)
    parser.add_argument('--min_exploration_rate', help='Minimum exploration rate for Q-learning',
                        type=float)

    parser.add_argument('--delta', help='PAC-probability (delta) for UCB-H and UCB-H+', type=float)
    parser.add_argument('--c', help='UCB-multiplier constant (c) for UCB-H and UCB-H+', type=float)

    parser.add_argument('--lambda', help='Exploration rate additive coefficient (lambda) for UCB-H+, see eq. (14)',
                        type=float, dest='lamb')
    parser.add_argument('--omega', help='Exploration rate power coefficient (omega) for UCB-H+, see eq. (14)',
                        type=float)

    parser.add_argument('-v', '--verbose', help='increase output verbosity', action='count')

    parser.add_argument('--save', help='Save the results into a csv-file', dest='save', action='store_true')
    parser.add_argument('--no-save', help='Do not save the results into a file', dest='save', action='store_false')
    parser.set_defaults(save=True)
    parser.add_argument('--save_dir', help='Directory to save the results', type=str)

    parser.add_argument('--plot', help='Plot the results with matplotlib', dest='plot', action='store_true')
    parser.add_argument('--no-plot', help='Do not plot the results', dest='plot', action='store_false')
    parser.set_defaults(plot=True)
    parser.add_argument('--smoothing', help='Percentage of episode to smooth plots over', type=float)
    parser.add_argument('--iqr', help='Interquantile range to plot, from 0.0 to 1.0', type=float)

    args = parser.parse_args()
    dict_args = vars(args)
    return dict_args


if __name__ == '__main__':
    # Parse the arguments
    args = parse_args()
    env_name = args['env']
    with open('defaults.yml') as f:
        defaults = yaml.safe_load(f)
    for key, value in args.items():
        if value is None:
            # print(f'"{key}" unspecified. Using the default value: {defaults[env_name][key]}')
            args[key] = defaults[env_name].get(key)

    # Run the actual script.
    run(**args)
