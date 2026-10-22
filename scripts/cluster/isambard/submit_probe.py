"""Submit a bounded, success-dependent capture -> fit -> evaluation Slurm chain."""

import argparse
import fcntl
import json
import os
import subprocess
import time
from contextlib import ExitStack
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True, help="Study directory on scratch")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--gpus", type=int, default=1, choices=range(1, 9))
    parser.add_argument("--cpus-per-task", type=int, default=4)
    parser.add_argument("--memory-per-node", required=True, help="Explicit Slurm --mem, e.g. 32G")
    parser.add_argument("--capture-time", required=True, help="Explicit walltime, e.g. 00:15:00")
    parser.add_argument("--train-time", required=True)
    parser.add_argument("--eval-time", required=True)
    parser.add_argument("--preset", default="pilot", choices=("smoke", "pilot", "task_vector"))
    parser.add_argument("overrides", nargs="*", help="Additional Hydra overrides")
    args = parser.parse_args()
    if args.cpus_per_task < 1:
        parser.error("cpus-per-task must be positive")
    protected = {
        "experiment.root",
        "probe.dataset.checkpoint",
        "execution.max_gpus",
        "device",
        "hydra.run.dir",
    }
    if any(item.lstrip("+").split("=")[0] in protected for item in args.overrides):
        parser.error("use the explicit root/checkpoint/GPU arguments; runtime/device are managed")
    repo = Path(__file__).resolve().parents[3]
    if Path.cwd().resolve() != repo:
        parser.error("submit from the repository root")
    root = args.root.expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    logs = root / "slurm"
    logs.mkdir(exist_ok=True)
    world_count = {"pilot": 1, "smoke": 2, "task_vector": 8}[args.preset]
    for override in args.overrides:
        if override.startswith("probe.dataset.num_worlds="):
            world_count = int(override.split("=", 1)[1])
    if world_count < args.gpus:
        parser.error("GPU workers must not exceed the number of worlds")
    scratch = Path(os.environ["SCRATCHDIR"])
    with ExitStack() as locks:
        for path in (scratch / ".iccl-probe-submission.lock", root / ".submission.lock"):
            guard = locks.enter_context(path.open("a"))
            fcntl.flock(guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
        record = root / "submission.json"
        if record.exists():
            previous = json.loads(record.read_text())
            ids = ",".join(previous["jobs"].values())
            if ids and subprocess.check_output(["squeue", "-h", "-j", ids], text=True).strip():
                parser.error("this study already has queued/running jobs; inspect submission.json")
        # Serialize probe chains for this account to keep their total GPU use at or below eight.
        queued = subprocess.check_output(
            ["squeue", "-h", "-u", os.environ["USER"], "-o", "%j"], text=True
        )
        if any(name.startswith("iccl-probe") for name in queued.splitlines()):
            parser.error("another probe job is queued/running; finish or cancel that chain first")
        common = [
            f"probe={args.preset}",
            f"experiment.root={root}",
            f"probe.dataset.checkpoint={args.checkpoint}",
            "device=cuda",
            f"execution.max_gpus={args.gpus}",
            *args.overrides,
        ]
        jobs: dict[str, str] = {}
        predecessor = None
        for stage, launcher, walltime in (
            ("capture", "capture_probe.slurm", args.capture_time),
            ("train", "train_probe.slurm", args.train_time),
            ("eval", "eval_probe.slurm", args.eval_time),
        ):
            command = [
                "sbatch",
                "--parsable",
                f"--nodes={(args.gpus + 3) // 4}",
                f"--ntasks={args.gpus}",
                f"--gpus={args.gpus}",
                f"--cpus-per-task={args.cpus_per_task}",
                f"--mem={args.memory_per_node}",
                f"--time={walltime}",
                f"--output={logs}/%x_%j.out",
                f"--error={logs}/%x_%j.err",
            ]
            if predecessor:
                command.extend([f"--dependency=afterok:{predecessor}", "--kill-on-invalid-dep=yes"])
            command.extend([str(Path("scripts/cluster/isambard") / launcher), *common])
            job = subprocess.check_output(command, text=True).strip().split(";")[0]
            if not job.isdigit():
                raise RuntimeError(f"unexpected sbatch result: {job}")
            jobs[stage] = predecessor = job
            # Persist each accepted job before attempting the next submission.
            record.write_text(
                json.dumps(
                    {
                        "jobs": jobs,
                        "gpus_per_stage": args.gpus,
                        "created": time.time(),
                        "arguments": common,
                    },
                    indent=2,
                )
                + "\n"
            )
            print(f"{stage}: {job}", flush=True)


if __name__ == "__main__":
    main()
