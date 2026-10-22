"""Exercise the submission CLI against a fake scheduler without allocating resources."""

import json
import os
import subprocess
import sys
from pathlib import Path


def test_submission_resources_dependencies_and_duplicate_guards(tmp_path):
    repo = Path(__file__).resolve().parents[1]
    commands = tmp_path / "bin"
    commands.mkdir()
    log = tmp_path / "submitted.jsonl"
    programs = {
        "squeue": "import os; print(os.environ.get('TEST_QUEUE', ''), end='')",
        "sbatch": (
            "import json, os, pathlib, sys\n"
            "p = pathlib.Path(os.environ['TEST_SUBMISSIONS'])\n"
            "rows = p.read_text().splitlines() if p.exists() else []\n"
            "with p.open('a') as stream: stream.write(json.dumps(sys.argv[1:]) + '\\n')\n"
            "print(1001 + len(rows))\n"
        ),
    }
    for name, source in programs.items():
        path = commands / name
        path.write_text(f"#!{Path(sys.executable).resolve()}\n{source}\n")
        path.chmod(0o755)
    environment = os.environ | {
        "PATH": str(commands) + os.pathsep + os.environ["PATH"],
        "TEST_SUBMISSIONS": str(log),
        "SCRATCHDIR": str(tmp_path),
    }
    command = [
        sys.executable,
        str(repo / "scripts/cluster/isambard/submit_probe.py"),
        "--root",
        str(tmp_path / "study"),
        "--checkpoint",
        "/tmp/frozen.pt",
        "--preset",
        "task_vector",
        "--gpus",
        "8",
        "--memory-per-node",
        "64G",
        "--capture-time",
        "00:10:00",
        "--train-time",
        "00:20:00",
        "--eval-time",
        "00:10:00",
    ]
    result = subprocess.run(command, cwd=repo, env=environment, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    rows = [json.loads(line) for line in log.read_text().splitlines()]
    assert len(rows) == 3
    assert all("--nodes=2" in row and "--gpus=8" in row and "--mem=64G" in row for row in rows)
    assert "--dependency=afterok:1001" in rows[1]
    assert "--dependency=afterok:1002" in rows[2]
    assert "--kill-on-invalid-dep=yes" in rows[1]
    repeated = subprocess.run(
        command,
        cwd=repo,
        env=environment | {"TEST_QUEUE": "RUNNING"},
        capture_output=True,
        text=True,
    )
    assert repeated.returncode != 0
    assert "queued/running" in repeated.stderr
    assert len(log.read_text().splitlines()) == 3
    over_cap = command.copy()
    over_cap[over_cap.index("--gpus") + 1] = "9"
    rejected = subprocess.run(over_cap, cwd=repo, env=environment, capture_output=True, text=True)
    assert rejected.returncode != 0
    assert len(log.read_text().splitlines()) == 3
