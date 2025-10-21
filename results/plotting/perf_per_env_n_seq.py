from results.common import *


def main(args: argparse.Namespace) -> None:
    plt.style.use('seaborn-v0_8-deep')
    plt.rcParams['axes.grid'] = True
    seeds, metric, sequences, methods = args.seeds, args.metric, args.sequences, args.methods
    n_envs = len(SEQUENCES[sequences[0]])
    fig, ax = plt.subplots(n_envs, 1, sharey='all', sharex='all', figsize=(11, 16))
    n_data_points = args.task_length * n_envs
    iterations = n_data_points * LOG_INTERVAL

    for i, sequence in enumerate(sequences):
        envs = SEQUENCES[sequence][:n_envs]
        colors = COLORS[sequence]
        for j, env in enumerate(envs):
            for k, method in enumerate(methods):
                data = load_data(env, n_data_points, method, metric, seeds, sequence, args.data_folder)
                plot_curve(ax[j], args.confidence, colors[k], f'{TRANSLATIONS[method]} ({sequence})', iterations, data,
                           len(seeds), linestyle=LINE_STYLES[i], interval=LOG_INTERVAL, sigma=4)

            ax[j].set_ylabel(TRANSLATIONS[metric])
            ax[j].set_title(TRANSLATIONS[env])
            ax[j].set_xlim([0, iterations])
            ax[j].set_ylim([0, 1])

    plot_name = "vs".join(sequences)
    n_col = len(methods) * len(sequences)
    if n_col > 8:
        n_col //= 2
    save_and_show(ax=ax[-1], plot_name=plot_name, n_col=n_col, vertical_anchor=-0.5)


if __name__ == "__main__":
    parser = common_plot_args()
    main(parser.parse_args())
