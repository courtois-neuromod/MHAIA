from results.common import *
from results.common import load_rl_baseline_data

COLOR_SAC = '#C44E52'
COLORS = ['#1F77B4', '#55A868', '#4C72B0', '#8172B2', '#CCB974', '#64B5CD', '#777777', '#FF8C00', '#917113']


def main(args: argparse.Namespace) -> None:
    plt.style.use('seaborn-v0_8-deep')
    seeds, sequence, methods, metric = args.seeds, args.sequence, args.methods, args.metric
    envs = SEQUENCES[sequence]
    n_envs = len(envs)
    n_methods = len(methods)
    fig_height = 1 + 1.5 * n_methods
    fig, ax = plt.subplots(n_methods, 1, sharex='all', sharey='all', figsize=(12, fig_height))
    task_length = args.task_length
    n_data_points = task_length * n_envs
    iterations = n_data_points * LOG_INTERVAL
    baseline = load_rl_baseline_data(sequence, seeds, task_length, args.data_folder, args.metric)
    baseline = gaussian_filter1d(baseline, sigma=KERNEL_SIGMA)
    results_dir = Path(__file__).parent.parent.resolve()

    for i, method in enumerate(methods):
        cur_ax = ax if n_methods == 1 else ax[i]
        seed_data = np.empty((len(seeds), int(iterations / LOG_INTERVAL)))
        seed_data[:] = np.nan
        for j, env in enumerate(envs):
            for k, seed in enumerate(seeds):
                path = results_dir / args.data_folder / sequence / method / f'seed_{seed}' / f'{env}_{metric}.json'
                if not os.path.exists(path):
                    continue
                with open(path, 'r') as f:
                    task_start = j * task_length
                    data = json.load(f)[task_start: task_start + task_length]
                    steps = len(data)
                    seed_data[k, np.arange(task_start, task_start + steps)] = data

        mean = np.nanmean(seed_data, axis=0)
        mean = gaussian_filter1d(mean, sigma=KERNEL_SIGMA)
        x = np.arange(0, iterations, LOG_INTERVAL)
        cur_ax.plot(x, mean, label=method, color=COLORS[i])
        cur_ax.plot(x, baseline, label='sac', color=COLOR_SAC)
        cur_ax.fill_between(x, mean, baseline, where=(mean < baseline), alpha=0.3, color=COLOR_SAC, interpolate=True)
        cur_ax.fill_between(x, mean, baseline, where=(mean >= baseline), alpha=0.3, color=COLORS[i], interpolate=True)

        if n_methods > 1:
            cur_ax.set_title(TRANSLATIONS[method], fontsize=13)
        cur_ax.set_xlim([0, iterations])
        cur_ax.set_ylim([0, 1])
        cur_ax.grid(True, which='major', axis='x', linestyle='--')

    top_ax = ax if n_methods == 1 else ax[0]
    bottom_ax = ax if n_methods == 1 else ax[-1]
    bottom_ax.ticklabel_format(style='sci', axis='x', scilimits=(0, 2))
    bottom_ax.tick_params(labelbottom=True)
    fontsize = 12 if n_methods > 1 else 10
    add_task_labels(top_ax, envs, iterations, n_envs, fontsize=fontsize)
    if n_methods > 1:
        main_ax = add_main_ax(fig, fontsize=13)
        main_ax.set_ylabel('Current Task Success', fontsize=13, labelpad=25)
    else:
        top_ax.set_ylabel('Current Task Success', fontsize=10)

    handles, labels = ax.get_legend_handles_labels() if n_methods == 1 else get_handles_and_labels(ax)
    sac_idx = labels.index('sac')
    handles.append(handles.pop(sac_idx))
    labels.append(labels.pop(sac_idx))
    labels = [TRANSLATIONS[label] for label in labels]

    n_cols = n_methods + 1
    bottom_ax.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, args.vertical_anchor), ncol=n_cols,
                     fancybox=True, shadow=True)
    file_name = sequence if not args.methods else f'{sequence}_{"_".join(methods)}'
    save_and_show(ax=bottom_ax, plot_name=file_name, add_xlabel=False, add_legend=False, h_pad=args.h_pad,
                  bottom_adjust=args.bottom_adjust)


def get_handles_and_labels(ax):
    handles, labels = [], []
    for a in ax:
        for h, l in zip(*a.get_legend_handles_labels()):
            if l not in labels:
                handles.append(h)
                labels.append(l)
    return handles, labels


if __name__ == "__main__":
    parser = common_plot_args()
    main(parser.parse_args())
