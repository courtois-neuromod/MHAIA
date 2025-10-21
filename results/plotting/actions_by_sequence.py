from results.common import *


def main(args: argparse.Namespace) -> None:
    plt.style.use('seaborn-v0_8-deep')
    plot_envs, seeds, method, sequences = args.test_envs, args.seeds, args.method, args.sequences
    envs = SEQUENCES[sequences[0]]
    n_envs, n_sequences = len(envs), len(sequences)
    short_sequence = is_short_sequence(envs)
    fig_width = 6 if short_sequence else 12
    fig_height = 2 + 2.75 * n_sequences
    fig_size = (fig_width, fig_height)
    iterations = args.task_length * n_envs
    cmap = plt.get_cmap('tab20c')

    if not plot_envs:
        plot_envs = ['train']

    for env in plot_envs:
        fig, ax = plt.subplots(n_sequences, 1, figsize=fig_size)
        folder = env if env == 'train' else f'test_{env}'

        for j, sequence in enumerate(sequences):
            data = load_action_data(folder, iterations, method, args.n_actions, seeds, sequence, args.data_folder)

            # Scale the values to add up to 1000 in each time step
            data = data / np.sum(data, axis=1, keepdims=True) * args.episode_length

            # Create a percent area stackplot with the values in mean
            sub_plot = ax if n_sequences == 1 else ax[j]
            sub_plot.stackplot(np.arange(iterations), data.T,
                               labels=[TRANSLATIONS[f'Action {i}'] for i in range(args.n_actions)],
                               colors=[cmap(i) for i in range(args.n_actions)])
            sub_plot.tick_params(labelbottom=True)
            sub_plot.set_title(sequence)
            sub_plot.set_ylabel("Actions/Episode", fontsize=14)
            sub_plot.set_xlim(0, iterations)
            sub_plot.set_ylim(0, args.episode_length)

        top_plot = ax if n_sequences == 1 else ax[0]
        add_task_labels(top_plot, envs, iterations, n_envs)

        bottom_plot = ax if n_sequences == 1 else ax[-1]
        bottom_plot.set_xlabel("Timesteps (K)", fontsize=14)
        n_cols = 4 if short_sequence else 3

        bottom_adjust = 0.12 if n_sequences > 1 else -0.125
        anchor_y = -0.625 if n_sequences > 1 else -0.6

        plt.tight_layout(rect=[0, bottom_adjust, 1, 1])
        bottom_plot.legend(loc='lower center', bbox_to_anchor=(0.5, anchor_y), ncol=n_cols, fancybox=True, shadow=True)

        file_path = '../plots/actions'
        os.makedirs(file_path, exist_ok=True)
        plt.savefig(f'{file_path}/{method}_{"_".join(sequences)}.pdf')
        plt.show()


if __name__ == "__main__":
    parser = common_action_plot_args()
    main(parser.parse_args())
