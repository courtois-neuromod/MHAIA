from results.common import *

LINE_STYLES = ['-', '--', ':', '-.']


def main(args: argparse.Namespace) -> None:
    plt.style.use('seaborn-v0_8-deep')
    plt.rcParams['axes.grid'] = True
    seeds, metric, sequences, methods = args.seeds, args.metric, args.sequences, args.methods
    example_sequence = sequences[0]
    colors = COLORS[example_sequence]
    envs = SEQUENCES[example_sequence]
    n_envs = len(envs)
    metric = None
    n_rows = 2
    n_cols = int(np.ceil(n_envs / n_rows))
    figure_width = 6 if example_sequence in ['CD4', 'CO4'] else 10
    fig, ax = plt.subplots(n_rows, n_cols, sharex='all', figsize=(figure_width, 4))
    max_steps = -np.inf
    task_length = args.task_length
    results_dir = Path(__file__).parent.parent.resolve()

    for i, env in enumerate(envs):
        row = i % n_cols
        col = i // n_cols
        for j, sequence in enumerate(sequences):
            for l, method in enumerate(args.methods):
                metric = args.metric if args.metric else METRICS[env]
                seed_data = np.empty((len(seeds), task_length))
                seed_data[:] = np.nan
                for k, seed in enumerate(seeds):
                    path = results_dir / args.data_folder / sequence / method / f'seed_{seed}' / f'{env}_{metric}.json'
                    if not os.path.exists(path):
                        print(f'Path {path} does not exist')
                        continue
                    with open(path, 'r') as f:
                        task_start = i * task_length
                        data = json.load(f)[task_start: task_start + task_length]
                    steps = len(data)
                    max_steps = max(max_steps, steps)
                    seed_data[k, np.arange(steps)] = data

                label = f'{TRANSLATIONS[method]} ({TRANSLATIONS[sequence]})'
                plot_curve(ax[col, row], args.confidence, colors[l], label, task_length, seed_data,
                           len(seeds), linestyle=LINE_STYLES[j])

        ax[col, row].set_ylabel(TRANSLATIONS[metric], fontsize=11)
        ax[col, row].set_title(TRANSLATIONS[env])
        ax[col, row].yaxis.set_label_coords(-0.25, 0.5)

    add_main_ax(fig, x_label='Timesteps (K)')
    plot_name = f'{len(methods)}_methods_{"vs".join(sequences)}'
    save_and_show(ax[-1, -1], plot_name=plot_name, n_col=len(methods) * 2, h_pad=args.h_pad, fig=fig, add_xlabel=False)


if __name__ == "__main__":
    parser = common_plot_args()
    main(parser.parse_args())
