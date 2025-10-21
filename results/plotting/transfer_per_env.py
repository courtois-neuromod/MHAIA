from results.common import *
from results.common import add_main_ax

COLORS = ['#C44E52', '#55A868']


def main(args: argparse.Namespace) -> None:
    plt.style.use('seaborn-v0_8-deep')
    plt.rcParams['axes.grid'] = True
    seeds = args.seeds
    sequences = ['single', args.sequence]
    envs = SEQUENCES[args.sequence]
    n_envs = len(envs)
    metric = args.metric
    n_rows = 2
    n_cols = int(np.ceil(n_envs / n_rows))
    figsize = (5, 4) if is_short_sequence(envs) else (11, 5)
    fig, ax = plt.subplots(n_rows, n_cols, sharex='all', figsize=figsize)
    task_length = args.task_length

    for i, env in enumerate(envs):
        row = i % n_cols
        col = i // n_cols
        reference = None
        for j, sequence in enumerate(sequences):
            task_start = 0 if sequence == 'single' else i * task_length
            method = 'sac' if sequence == 'single' else args.method
            data_file = env if sequence == 'single' else 'train'
            data = load_data(data_file, task_length, method, metric, seeds, sequence, args.data_folder, task_start)

            mean = np.nanmean(data, axis=0)
            mean = gaussian_filter1d(mean, sigma=2)

            ax[col, row].plot(mean, label=TRANSLATIONS[method], color=COLORS[j])
            ax[col, row].tick_params(labelbottom=True)
            ax[col, row].ticklabel_format(style='sci', axis='y', scilimits=(0, 4))
            if reference is None:
                reference = mean
            else:
                ax[col, row].fill_between(np.arange(task_length), mean, reference, where=(mean < reference), alpha=0.2,
                                          color=COLORS[0], interpolate=True)
                ax[col, row].fill_between(np.arange(task_length), mean, reference, where=(mean >= reference), alpha=0.2,
                                          color=COLORS[1], interpolate=True)

        ax[col, row].set_ylabel(TRANSLATIONS[metric], fontsize=9)
        ax[col, row].set_title(TRANSLATIONS[env], fontsize=11)
        ax[col, row].yaxis.set_label_coords(-0.27, 0.5)

    add_main_ax(fig, x_label='Timesteps (K)')
    plot_name = f'{args.sequence}_{args.method}_individual'
    n_col = 2  # RL baseline and CL method
    save_and_show(ax[-1, -1], plot_name=plot_name, n_col=n_col, h_pad=args.h_pad, add_xlabel=False, fig=fig)


if __name__ == "__main__":
    parser = common_plot_args()
    main(parser.parse_args())
