from results.common import *


def main(args: argparse.Namespace) -> None:
    plt.style.use('seaborn-v0_8-deep')
    seeds = args.seeds
    sequence = args.sequence
    envs = SEQUENCES[sequence]
    n_envs = len(envs)
    metric = None
    short_sequence = is_short_sequence(sequence)
    fig_width = 10
    fig_height = 8 if short_sequence else 13
    fig_size = (fig_width, fig_height)
    y_label_shift = -0.06 if short_sequence else -0.04
    share_y = sequence in ['CD4', 'CD8'] or args.metric == 'success'
    fig, ax = plt.subplots(n_envs, 1, sharex='all', sharey=share_y, figsize=fig_size)
    methods = args.methods
    n_data_points = args.task_length * n_envs
    iterations = n_data_points * LOG_INTERVAL

    for i, env in enumerate(envs):
        for j, method in enumerate(methods):
            metric = args.metric if args.metric else METRICS[env] if env in METRICS else 'kills'
            data = load_data(env, n_data_points, method, metric, seeds, sequence, args.data_folder)
            plot_curve(ax[i], args.confidence, PLOT_COLORS[j], TRANSLATIONS[method], iterations, data, len(seeds),
                       interval=LOG_INTERVAL)

        ax[i].set_ylabel(TRANSLATIONS[metric], fontsize=11)
        ax[i].set_title(TRANSLATIONS[env], fontsize=11)
        ax[i].set_xlim([0, iterations])
        ax[i].set_ylim([0, 1])
        ax[i].yaxis.set_label_coords(y_label_shift, 0.5)

    add_coloured_task_labels(ax[0], sequence, iterations)
    fontsize = 11
    vertical_anchor = -0.7 if short_sequence else -0.8
    plot_name = f'{sequence}_{metric}'
    n_col = len(methods)
    save_and_show(ax=ax[-1], plot_name=plot_name, n_col=n_col, vertical_anchor=vertical_anchor, fontsize=fontsize)


if __name__ == "__main__":
    parser = common_plot_args()
    main(parser.parse_args())
