from results.common import *


def main(args: argparse.Namespace) -> None:
    plt.style.use('seaborn-v0_8-deep')
    plt.rcParams['axes.grid'] = True
    seeds, metric, sequences, methods = args.seeds, args.metric, args.sequences, args.methods
    n_sequences, n_seeds = len(sequences), len(seeds)
    fig_size = (12, 7) if n_sequences > 1 else (10, 3)
    fig, axes = plt.subplots(n_sequences, 1, sharey='all', sharex='all', figsize=fig_size)
    assert len(sequences) > 0, "No sequences provided"

    for i, sequence in enumerate(sequences):
        ax = axes if n_sequences == 1 else axes[i]
        envs = SEQUENCES[sequence]
        n_envs = len(envs)
        iterations = args.task_length * n_envs * LOG_INTERVAL
        n_data_points = args.task_length * n_envs
        for j, method in enumerate(methods):
            data = load_data_per_env(envs, n_data_points, method, metric, seeds, sequence, args.data_folder)
            plot_curve(ax, args.confidence, PLOT_COLORS[j], TRANSLATIONS[method], iterations, data, n_seeds * n_envs,
                       agg_axes=(0, 1), interval=LOG_INTERVAL)

        ax.set_ylabel('Average Success', fontsize=11)
        ax.set_xlim([0, iterations])
        ax.set_ylim([0, 1])
        if n_sequences > 1:
            ax.set_title(sequence, fontsize=14)
        add_task_labels(ax, envs, iterations, n_envs, fontsize=9)

    vertical_anchor = -0.45
    bottom_adjust = 0 if n_sequences > 1 else -0.1
    plot_name = f'success_{"_".join(sequences)}'
    ax_last = axes[-1] if n_sequences > 1 else axes
    save_and_show(ax=ax_last, plot_name=plot_name, n_col=len(methods), vertical_anchor=vertical_anchor,
                  bottom_adjust=bottom_adjust)


if __name__ == "__main__":
    parser = common_plot_args()
    main(parser.parse_args())
