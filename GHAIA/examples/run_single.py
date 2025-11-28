import argparse

from GHAIA.env.builder import make_env
from GHAIA.utils.config import Scenario


def main(args: argparse.Namespace):
    scenario = Scenario[args.scenario.upper()]
    env = make_env(scenario, args.task)
    env.reset()
    total_reward = 0
    success = 0
    for steps in range(args.max_steps):
        action = env.action_space.sample()
        state, reward, done, truncated, info = env.step(action)
        total_reward += reward
        success += env.get_success()
        if args.render:
            env.render()
        if done or truncated:
            break
    print(f"Task {env.task_id}-{env.name} finished after {steps} steps.")
    print(f"Total Reward: {total_reward:.2f}")
    print(f"Average Success: {success / (steps + 1):.2f}")
    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Continual Super Mario Bros - Single Environment Test")
    parser.add_argument('--scenario', type=str, default='world1',
                        choices=['world1', 'world2', 'world3', 'world4',
                                 'world5', 'world6', 'world7', 'world8'],
                        help="Which world to run")
    parser.add_argument("--task", type=str, default='Level1-1',
                        help="Name of the level to run (e.g., Level1-1, Level2-3)")
    parser.add_argument("--max-steps", type=int, default=5000,
                        help="Maximum number of steps to run")
    parser.add_argument("--render", action='store_true', default=False,
                        help="Whether to render the environment")
    main(parser.parse_args())
