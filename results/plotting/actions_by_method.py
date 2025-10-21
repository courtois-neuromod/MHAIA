from results.common import *
from results.common import load_action_data


def main(args: argparse.Namespace) -> None:
    plt.style.use('seaborn-v0_8-deep')
    plot_envs, seeds, methods, sequence = args.test_envs, args.seeds, args.methods, args.sequence
    envs = SEQUENCES[sequence]
    n_methods, n_envs = len(methods), len(envs)
    short_sequence = is_short_sequence(sequence)
    fig_width = 6 if short_sequence else 12
    fig_height = 3 + 2.5 * n_methods
    fig_size = (fig_width, fig_height)
    cmap = plt.get_cmap('tab20c')
    iterations = args.task_length * n_envs

    if not plot_envs:
        plot_envs = ['train']

    for env in plot_envs:
        fig, ax = plt.subplots(n_methods, 1, figsize=fig_size)
        folder = env if env == 'train' else f'test_{env}'

        for j, method in enumerate(methods):
            data = load_action_data(folder, iterations, method, args.n_actions, seeds, sequence, args.data_folder)

            # Create a percent area stackplot with the values in mean
            sub_plot = ax if n_methods == 1 else ax[j]
            sub_plot.stackplot(np.arange(iterations), data.T,
                         labels=[TRANSLATIONS[f'Action {i}'] for i in range(args.n_actions)],
                         colors=[cmap(i) for i in range(args.n_actions)])
            sub_plot.tick_params(labelbottom=True)
            sub_plot.set_title(TRANSLATIONS[method])
            sub_plot.set_ylabel("Actions/Episode")
            sub_plot.set_xlim(0, iterations)
            sub_plot.set_ylim(0, args.episode_length)

        top_plot = ax if n_methods == 1 else ax[0]
        add_task_labels(top_plot, envs, iterations, n_envs)

        title = env.capitalize() if env == 'train' else TRANSLATIONS[SEQUENCES[sequence][env]]

        bottom_plot = ax if n_methods == 1 else ax[-1]
        bottom_plot.set_xlabel("Timesteps (K)", fontsize=11)
        n_cols = 4 if short_sequence else 3

        bottom_adjust = -0.031 + 0.21 / (n_methods ** 0.5)
        anchor_y = -0.78 + 0.28 / (n_methods ** 0.7)

        plt.tight_layout(rect=[0, bottom_adjust, 1, 1])
        bottom_plot.legend(loc='lower center', bbox_to_anchor=(0.5, anchor_y), ncol=n_cols, fancybox=True, shadow=True)

        file_path = '../plots/actions'
        os.makedirs(file_path, exist_ok=True)
        plt.savefig(f'{file_path}/{sequence}_{title}.pdf')
        plt.show()


if __name__ == "__main__":
    parser = common_action_plot_args()
    main(parser.parse_args())
