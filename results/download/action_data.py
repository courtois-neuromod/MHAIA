import wandb
from wandb.apis.public import Run

from results.common import *


def main(args: argparse.Namespace) -> None:
    api = wandb.Api()
    runs = api.runs(args.project)
    base_dir = Path(__file__).parent.parent.resolve()
    for run in runs:
        if suitable_run(run, args):
            store_data(base_dir, run, args.sequence, args.test_envs, args.eval_mode, args.n_actions, args.data_folder)


def suitable_run(run, args: argparse.Namespace) -> bool:
    # Check whether the run shouldn't be filtered out
    if any(logs in run.name for logs in args.include_runs):
        return True
    # Check whether the run has successfully finished
    if run.state != "finished":
        return False
    # Load the configuration of the run
    config = run.config
    # Check whether the provided CL sequence corresponds to the run
    if args.sequence not in run.url:
        return False
    # Check whether the wandb tags are suitable
    if 'wandb_tags' in config:
        tags = config['wandb_tags']
        if any(tag in tags for tag in FORBIDDEN_TAGS):
            return False
    # Check whether the run corresponds to one of the provided seeds
    if args.seeds:
        if 'seed' not in config:
            return False
        seed = config['seed']
        if seed not in args.seeds:
            return False
    if args.method:
        method = get_cl_method(run)
        if method != args.method:
            return False
    # All filters have been passed
    return True


def store_data_for_env(base_dir: Path, run: Run, sequence: str, eval_mode: str, n_actions: int, data_folder: str,
                       test_env: int = None) -> None:
    if test_env is not None:
        task = SEQUENCES[sequence][test_env]
        env = f'run_and_gun-{task}' if sequence in ['CD4', 'CD8'] else f'{task}-{ENVS[sequence]}'
        log_key = f'test/{eval_mode}/{test_env}/{env}/actions'
    else:
        log_key = 'train/actions'
    log_keys = [f'{log_key}/{i}' for i in range(n_actions)]
    history = list(iter(run.scan_history(keys=log_keys)))
    if not history:
        return
    actions = np.array([[log[f'{log_key}/{i}'] for i in range(n_actions)] for log in history])

    method = get_cl_method(run)
    seed = run.config["seed"]
    folder = 'train' if test_env is None else f'test_{test_env}'
    path = base_dir / data_folder / 'actions' / sequence / method / folder
    os.makedirs(path, exist_ok=True)

    file_path = path / f'seed_{seed}.json'
    print(f'Saving {run.id} run actions to {file_path}')
    with open(file_path, 'w') as f:
        json.dump(actions.tolist(), f)


def store_data(base_dir: Path, run: Run, sequence: str, test_envs: List[int], eval_mode: str, n_actions: int,
               data_folder: str) -> None:
    for env in (test_envs or [None]):
        store_data_for_env(base_dir, run, sequence, eval_mode, n_actions, data_folder, env)


def action_dl_args() -> argparse.ArgumentParser:
    parser = common_dl_args()
    parser.add_argument("--n_actions", type=int, default=12,
                        help="Number of discrete actions that the models were trained with")
    return parser


if __name__ == "__main__":
    parser = action_dl_args()
    main(parser.parse_args())
