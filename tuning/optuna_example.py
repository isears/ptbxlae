import optuna
import argparse


def objective(trial):
    x = trial.suggest_float("x", -10, 10)
    return (x - 2) ** 2


def parse_args():
    parser = argparse.ArgumentParser(
        description="Lightweight optuna example to troubleshoot distributed training"
    )

    parser.add_argument(
        "--timelimit",
        type=float,
        default=10.0,
        help="Time limit for slurm jobs in minutes",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    study = optuna.load_study(
        study_name="distributed-test", storage="sqlite:///cache/distributed-test.db"
    )
    study.optimize(objective, timeout=(args.timelimit * 60))
