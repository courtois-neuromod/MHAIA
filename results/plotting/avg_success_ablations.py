from results.common import *


def main(args: argparse.Namespace) -> None:
    plt.style.use('seaborn-v0_8-deep')
    plt.rcParams['axes.grid'] = True
    methods, seeds, ablations, sequence, metric = args.methods, args.seeds, args.ablations, args.sequence, args.metric
    envs = SEQUENCES[sequence]
    n_envs, n_seeds, n_methods = len(envs), len(seeds), len(methods)
    fig_width = 12
    fig_height = 1 + 2 * n_methods
    fig_size = (fig_width, fig_height)
    fig, ax = plt.subplots(n_methods, 1, sharey='all', sharex='all', figsize=fig_size)
    iterations = args.task_length * n_envs

    for i, method in enumerate(methods):
        ax_to_plot = ax[i] if n_methods > 1 else ax
        for j, tag in enumerate(ablations):
            data = load_data_per_env(envs, iterations, method, metric, seeds, sequence, args.data_folder, tag)
            plot_curve(ax_to_plot, args.confidence, PLOT_COLORS[j], TRANSLATIONS[tag], iterations, data,
                       n_seeds * n_envs, agg_axes=(0, 1))

        ax_to_plot.set_ylabel('Average Success')
        ax_to_plot.set_title(TRANSLATIONS[method])
        ax_to_plot.set_xlim(0, iterations)
        ax_to_plot.set_ylim(0, 1)

    ax_to_add_labels = ax[0] if n_methods > 1 else ax
    add_task_labels(ax_to_add_labels, envs, iterations, n_envs)
    ax_last = ax[-1] if n_methods > 1 else ax

    save_and_show(ax_last, args.plot_name, len(ablations), vertical_anchor=args.vertical_anchor, h_pad=args.h_pad,
                  bottom_adjust=args.bottom_adjust)


if __name__ == "__main__":
    parser = common_plot_args()
    parser.add_argument("--ablations", type=str, default=['multi_head', 'single_head'], nargs='+',
                        help="Data folder names of the ablations (e.g., no_reg_critic no_task_id multi_head")
    parser.add_argument("--plot_name", type=str, default="ablation_curves",
                        help="Optional name of the plot (useful when comparing ablations)")
    main(parser.parse_args())
