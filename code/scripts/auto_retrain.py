from __future__ import annotations

import argparse
import os
import subprocess
from datetime import datetime
from pathlib import Path


def _run(cmd: list[str], env: dict | None = None) -> None:
    print(" ".join(cmd))
    subprocess.run(cmd, check=True, env=env)


def main() -> None:
    parser = argparse.ArgumentParser(description="Automated retrain + pipeline + monitoring")
    parser.add_argument("--raw_csv", required=True)
    parser.add_argument("--external_dir", required=True)
    parser.add_argument("--use_external", action="store_true")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--omp_threads", type=int, default=1)
    parser.add_argument("--model_strategy", choices=["lgbm", "ensemble", "auto"], default="lgbm")
    parser.add_argument("--ensemble_tune", action="store_true")
    parser.add_argument("--ensemble_trials", type=int, default=30)
    parser.add_argument("--ensemble_timeout", type=int, default=300)
    parser.add_argument("--ensemble_fs", action="store_true")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = str(args.omp_threads)

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"Run ID: {run_id}")

    # 1) Deep learning retrain (순유입)
    _run(
        [
            "python3",
            str(repo_root / "scripts" / "deep_learning_models.py"),
            "--raw_csv",
            args.raw_csv,
            "--external_dir",
            args.external_dir,
            "--epochs",
            str(args.epochs),
        ],
        env=env,
    )

    # 2) Main pipeline
    cmd = [
        "python3",
        str(repo_root / "scripts" / "run_mvp.py"),
        "--raw_csv",
        args.raw_csv,
        "--encoding",
        "cp949",
        "--external_dir",
        args.external_dir,
        "--model_strategy",
        args.model_strategy,
    ]
    if args.use_external:
        cmd.append("--use_external")
    if args.ensemble_tune:
        cmd.append("--ensemble_tune")
    if args.ensemble_trials:
        cmd += ["--ensemble_trials", str(args.ensemble_trials)]
    if args.ensemble_timeout:
        cmd += ["--ensemble_timeout", str(args.ensemble_timeout)]
    if args.ensemble_fs:
        cmd.append("--ensemble_fs")

    _run(cmd, env=env)

    # 3) Model evaluation (refresh selection CSV)
    _run(
        [
            "python3",
            str(repo_root / "scripts" / "evaluate_models.py"),
            "--panel_path",
            str(repo_root / "outputs" / "segment_panel.parquet"),
            "--model_dir",
            str(repo_root / "outputs" / "models"),
            "--out_csv",
            str(repo_root / "outputs" / "model_evaluation.csv"),
        ],
        env=env,
    )

    # 4) Monitoring report
    _run(
        [
            "python3",
            str(repo_root / "scripts" / "monitoring_report.py"),
        ],
        env=env,
    )


if __name__ == "__main__":
    main()
