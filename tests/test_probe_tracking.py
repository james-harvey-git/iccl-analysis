import sys
from pathlib import Path
from types import SimpleNamespace

from iccl.analysis.probe_tracking import StudyReporter, WorkerEvents


def test_offline_replay_single_identity_and_deduplicated_restart(probe_cfg, monkeypatch):
    cfg = probe_cfg
    root = Path(cfg.experiment.root) / cfg.experiment.phase
    events = WorkerEvents(root, "train", "first", 0)
    events.emit("fit_complete", world=0, fit_id="example")
    events.close()
    offline = StudyReporter(cfg, root)
    offline.flush()
    offline.finish(success=True)
    sent, identities = [], []

    class Run:
        step = 0
        config = SimpleNamespace(update=lambda *a, **k: None)

        def log(self, payload, *, step):
            assert step >= self.step
            sent.append(payload)
            self.step = step + 1

        def finish(self, **kwargs):
            pass

    run = Run()

    def initialize(**kwargs):
        identities.append(kwargs["id"])
        return run

    monkeypatch.setitem(sys.modules, "wandb", SimpleNamespace(init=initialize))
    cfg.wandb.mode = "online"
    first = StudyReporter(cfg, root)
    first.flush()
    first.flush()
    first.finish(success=True)
    assert len(sent) == 1
    events = WorkerEvents(root, "eval", "second", 0)
    events.emit("world_complete", world=0)
    events.close()
    second = StudyReporter(cfg, root)
    second.flush()
    second.finish(success=True)
    assert len(sent) == 2
    assert len(set(identities)) == 1
    assert sent[-1]["progress/completed_fits"] == 1
    assert sent[0]["phase"] == "development"
