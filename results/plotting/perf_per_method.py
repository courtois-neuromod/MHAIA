from results.common import *


def main(args: argparse.Namespace) -> None:
    plt.style.use('seaborn-v0_8-deep')
    seeds, metric, sequence = args.seeds, args.metric, args.sequence
    colors = COLORS[sequence]
    envs = SEQUENCES[sequence]
    n_seeds, n_envs = len(seeds), len(envs)
    methods = args.methods
    n_methods = len(methods)
    fig_width = 12
    fig_height = 1 + 1.5 * n_methods
    fig_size = (fig_width, fig_height)
    fig, ax = plt.subplots(len(methods), 1, sharey='all', sharex='all', figsize=fig_size)
    n_data_points = args.task_length * n_envs
    iterations = n_data_points * LOG_INTERVAL

    for i, method in enumerate(methods):
        cur_ax = ax if n_methods == 1 else ax[i]
        for j, env in enumerate(envs):
            data = load_data(env, n_data_points, method, metric, seeds, sequence, args.data_folder)
            plot_curve(cur_ax, args.confidence, colors[j], TRANSLATIONS[env], iterations, data, n_seeds,
                       interval=LOG_INTERVAL, sigma=4)
        if n_methods > 1:
            cur_ax.set_title(TRANSLATIONS[method], fontsize=12)
        cur_ax.set_ylabel(TRANSLATIONS[metric])
        cur_ax.set_xlim([0, iterations])
        cur_ax.set_ylim([0, 1])

    top_ax = ax if n_methods == 1 else ax[0]
    bottom_ax = ax if n_methods == 1 else ax[-1]
    add_coloured_task_labels(top_ax, sequence, iterations, fontsize=9)
    method = f'_{methods[0]}' if n_methods == 1 else ''
    plot_name = f'{sequence}_{metric}{method}'
    save_and_show(ax=bottom_ax, plot_name=plot_name, add_xlabel=n_methods > 1, add_legend=False)


if __name__ == "__main__":
    parser = common_plot_args()
    main(parser.parse_args())
