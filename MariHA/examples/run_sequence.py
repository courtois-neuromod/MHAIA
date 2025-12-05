import argparse

from MHAIA.env.continual import ContinualLearningEnv
from MHAIA.utils.config import Sequence


def main(args: argparse.Namespace):
    sequence = Sequence[args.sequence.upper()]
    cl_env = ContinualLearningEnv(sequence, steps_per_env=args.steps_per_env)

    print(f"Running Continual Learning Sequence: {args.sequence}")
    print(f"Number of tasks: {len(cl_env.tasks)}")
    print(f"Steps per environment: {args.steps_per_env}")
    print()

    for env in cl_env.tasks:
        env.reset()
        total_reward = 0
        success = 0
        for steps in range(args.steps_per_env):
            action = env.action_space.sample()
            state, reward, done, truncated, info = env.step(action)
            total_reward += reward
            success += env.get_success()
            if args.render:
                env.render()
            if done or truncated:
                env.reset()
        print(f"Task {env.task_id}-{env.name} finished.")
        print(f"  Total Reward: {total_reward:.2f}")
        print(f"  Average Success: {success / args.steps_per_env:.3f}")
        print()
        env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Continual Super Mario Bros - Sequence Test")
    parser.add_argument("--sequence", type=str, default='WORLD_PROGRESSION_4',
                        choices=['WORLD_PROGRESSION_4', 'WORLD_PROGRESSION_8', 'STAGE_TYPES_4',
                                 'WORLD_COMPLETE', 'DIFFICULTY_CURVE', 'MIXED_WORLDS'],
                        help="Name of the continual learning sequence")
    parser.add_argument("--steps-per-env", type=int, default=1000,
                        help="Number of steps to run in each environment")
    parser.add_argument("--render", action='store_true', default=False,
                        help="Whether to render the environments")
    main(parser.parse_args())
