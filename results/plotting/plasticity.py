from results.common import *

# Logging of some metrics begins when the corresponding task starts
offsets = {
    'CO4': [0, 0, 0, 3],
    'CO8': [0, 1, 0, 0, 4, 0, 4, 2]
}


def main(args: argparse.Namespace) -> None:
    plt.style.use('seaborn-v0_8-deep')
    plt.rcParams['axes.grid'] = True
    seeds, method, sequence, n_repeats, task_length = args.seeds, args.method, args.sequence, args.n_repeats, args.task_length
    envs = SEQUENCES[args.sequence]
    n_envs = len(envs)
    n_seeds = len(seeds)
    short_sequence = is_short_sequence(envs)
    n_rows = 1 if short_sequence else 2
    n_cols = int(np.ceil(n_envs / n_rows))
    figsize = (10, 2.5) if short_sequence else (11, 5)
    fig, ax = plt.subplots(n_rows, n_cols, sharex='all', figsize=figsize)
    n_data_points = task_length * n_envs * n_repeats
    results_dir = Path(__file__).parent.parent.resolve()

    for i, env in enumerate(envs):
        row = i % n_cols
        col = i // n_cols
        metric = METRICS[env]
        seed_data = np.empty((n_seeds, task_length * n_repeats))
        seed_data[:] = np.nan
        for k, seed in enumerate(seeds):
            path = results_dir / args.data_folder / f'repeat_{n_repeats}' / sequence / method / f'seed_{seed}' / f'train_{metric}.json'
            if not os.path.exists(path):
                print(f'Path {path} does not exist')
                continue
            with open(path, 'r') as f:
                data = np.array(json.load(f))
            task_start = i * task_length - offsets[sequence][i] * task_length
            start_time_steps = np.arange(task_start, n_data_points, n_envs * task_length)
            start_time_steps = start_time_steps[start_time_steps < len(data)]  # In case of early stopping
            data = [data[env_data_start_point: env_data_start_point + task_length] for env_data_start_point in
                    start_time_steps]
            data = np.concatenate(data)  # Concatenate data from all repeats
            steps = len(data)
            seed_data[k, np.arange(steps)] = data

        iterations = task_length * n_repeats * LOG_INTERVAL
        cur_ax = ax[col, row] if n_rows > 1 else ax[row]
        plot_curve(cur_ax, args.confidence, PLOT_COLORS[0], TRANSLATIONS[method], iterations, seed_data, len(seeds),
                   interval=LOG_INTERVAL)

        add_main_ax(fig, fontsize=9, labelpad=30, x_label='# of Training steps (100K for Each Task)')
        cur_ax.set_ylabel(TRANSLATIONS[metric], fontsize=9)
        cur_ax.ticklabel_format(axis='y', style='sci', scilimits=(0, 4), useMathText=True)
        cur_ax.set_title(TRANSLATIONS[env], fontsize=11)

    bottom_adjust = -0.22 if is_short_sequence(envs) else -0.1
    plot_name = f'plasticity_{sequence}'
    save_and_show(ax[-1, -1], plot_name=plot_name, bottom_adjust=bottom_adjust, add_xlabel=False, add_legend=False)


if __name__ == "__main__":
    parser = common_plot_args()
    parser.add_argument("--n_repeats", type=int, default=10, help="Number of task sequence repeats")
    main(parser.parse_args())
