from results.common import *


def main(args: argparse.Namespace) -> None:
    methods, seeds, tags, sequence, metric, task_length, confidence = \
        args.methods, args.seeds, args.tags, args.sequence, args.metric, args.task_length, args.confidence
    ablation_tags = [t for t in tags if t != 'default']
    n_seeds, n_methods, n_tags, n_ablations = len(seeds), len(methods), len(tags), len(ablation_tags)

    data = np.empty((len(methods), n_ablations))
    data[:] = np.nan
    default = None

    for i, tag in enumerate(tags):
        cl_data, ci_data, transfer_data = load_cl_data(methods, metric, seeds, sequence, args.data_folder, task_length,
                                                       confidence, tag=tag)
        performance = calculate_performance(cl_data)
        if any(np.isnan(performance)):
            print(f'Warning: NaN values found in performance for {tag}. Skipping this ablation.')
            continue
        if tag == 'default':
            default = performance
        else:
            col = ablation_tags.index(tag)
            data[:, col] = performance

    plot_histograms(data, default, tags, methods)


def plot_histograms(data: ndarray, default: ndarray, ablations: List[str], methods: List[str]):
    plt.style.use('seaborn-v0_8-deep')
    figsize = (10, 2)
    fig, axes = plt.subplots(1, len(ablations) - 1, sharey='all', figsize=figsize)
    ablations = ablations[1:]  # Remove the original data folder

    for i, variation in enumerate(ablations):
        for j, method in enumerate(methods):
            diff = ((data[j, i] - default[j]) / default[j]) * 100
            axes[i].bar(i + j, [diff], label=TRANSLATIONS[method], color=METHOD_COLORS[method])

        axes[i].axhline(0, color='black', lw=1)
        variation = TRANSLATIONS[variation]
        axes[i].set_title(f'{variation}', fontsize=11)
        axes[i].set_xticks([])

    def format_percent(x, pos):
        return f"{x:.0f}%"  # Format the y-labels with a percentage sign

    # Apply the formatting function to the y-labels
    axes[0].yaxis.set_major_formatter(plt.FuncFormatter(format_percent))
    axes[0].set_ylabel('Performance Increase')
    plt.ylim(-100, 100)
    save_and_show(axes[-1], plot_name='variations', n_col=len(methods), bottom_adjust=0.1, fig=fig, add_xlabel=False)


if __name__ == "__main__":
    parser = common_plot_args()
    parser.add_argument("--tags", type=str, required=True, nargs='+', help="Names of the wandb tags")
    main(parser.parse_args())
