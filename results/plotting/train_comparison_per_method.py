from results.common import *


def main(args: argparse.Namespace) -> None:
    plt.style.use('seaborn-v0_8-deep')
    plt.rcParams['axes.grid'] = True
    seeds, metric, methods, sequences = args.seeds, args.metric, args.methods, args.sequences
    envs = SEQUENCES[sequences[0]]
    n_envs = len(envs)
    n_methods = len(methods)
    fig_height = 1 + 2 * n_methods
    fig, ax = plt.subplots(n_methods, 1, sharey='all', sharex='all', figsize=(12, fig_height))
    iterations = args.task_length * n_envs

    for i, method in enumerate(methods):
        ax_to_plot = ax[i] if n_methods > 1 else ax
        for s, sequence in enumerate(sequences):
            envs = SEQUENCES[sequence]
            envs = envs[:n_envs]
            colors = COLORS[sequence]
            data = load_data('train', iterations, method, metric, seeds, sequence, args.data_folder)
            plot_curve(ax_to_plot, args.confidence, colors[s], sequence, iterations, data, len(seeds))

        ax_to_plot.set_ylabel(TRANSLATIONS[metric])
        ax_to_plot.set_title(TRANSLATIONS[method])
        ax_to_plot.set_xlim(0, iterations)
        ax_to_plot.set_ylim(0, 1)

    first_ax = ax[0] if n_methods > 1 else ax
    last_ax = ax[-1] if n_methods > 1 else ax
    add_task_labels(first_ax, envs, iterations, n_envs)
    plot_name = f'{"vs".join(sequences)}_{n_methods}_methods'
    save_and_show(ax=last_ax, plot_name=plot_name, n_col=len(sequences), vertical_anchor=args.vertical_anchor,
                  bottom_adjust=args.bottom_adjust, h_pad=args.h_pad)


if __name__ == "__main__":
    parser = common_plot_args()
    main(parser.parse_args())
