"""Exports the frozen evaluation suites defined by the data config.

Suites: "in_dist" (standard curriculum sequences), "composite" (curriculum plus
a few-shot composite final task) with its paired "composite_control" (same
world and final task, no history), one "structural_<graph>" suite per
overlap-graph family in eval_sets.structural_graphs, and "retention"
(curriculum sequences that re-demonstrate the first task at the end, for the
forgetting/savings metrics). All suites carry their worlds as interp probe
targets.
"""

from dataclasses import replace
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

from iccl.data.dataset import make_family, sequence_config_from, sequence_rng
from iccl.data.export import export_suite
from iccl.data.sequences import FinalTaskConfig, build_paired_control, build_sequence

EVAL_SEED_OFFSET = 1_000_000  # keeps eval streams disjoint from training indices


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    data_cfg = cfg.data
    composite_cfg = data_cfg.eval_sets.composite
    family = make_family(data_cfg, extra_hotness=composite_cfg.hotness)
    seq_cfg = sequence_config_from(data_cfg)
    final = FinalTaskConfig(
        mode="composite", hotness=composite_cfg.hotness, num_demos=composite_cfg.num_demos
    )
    out_dir = Path(data_cfg.eval_sets.out_dir)
    num = data_cfg.eval_sets.num_sequences
    meta = {"config": OmegaConf.to_container(data_cfg, resolve=True), "seed": cfg.seed}

    in_dist, composite, control = [], [], []
    for i in range(num):
        rng = sequence_rng(cfg.seed + EVAL_SEED_OFFSET, i)
        in_dist.append(build_sequence(family, seq_cfg, rng, include_world=True))
        rng = sequence_rng(cfg.seed + EVAL_SEED_OFFSET, num + i)
        seq = build_sequence(family, seq_cfg, rng, final_task=final, include_world=True)
        composite.append(seq)
        control.append(build_paired_control(family, seq_cfg, seq, final, rng))

    export_suite(in_dist, out_dir / "in_dist", dict(meta, suite="in_dist"))
    export_suite(composite, out_dir / "composite", dict(meta, suite="composite"))
    export_suite(control, out_dir / "composite_control", dict(meta, suite="composite_control"))

    num_suites = 3
    for graph_offset, graph in enumerate(data_cfg.eval_sets.structural_graphs):
        graph_cfg = replace(seq_cfg, task_graph=graph)
        suite = [
            build_sequence(
                family,
                graph_cfg,
                sequence_rng(cfg.seed + EVAL_SEED_OFFSET, (2 + graph_offset) * num + i),
                include_world=True,
            )
            for i in range(num)
        ]
        suite_meta = dict(meta, suite=f"structural_{graph}")
        export_suite(suite, out_dir / f"structural_{graph}", suite_meta)
        num_suites += 1

    revisit_demos = data_cfg.eval_sets.retention.revisit_demos
    retention_base = (2 + len(data_cfg.eval_sets.structural_graphs)) * num
    retention = [
        build_sequence(
            family,
            seq_cfg,
            sequence_rng(cfg.seed + EVAL_SEED_OFFSET, retention_base + i),
            revisit_demos=revisit_demos,
            include_world=True,
        )
        for i in range(num)
    ]
    export_suite(retention, out_dir / "retention", dict(meta, suite="retention"))
    num_suites += 1
    print(f"exported {num_suites} suites x {num} sequences to {out_dir}/")


if __name__ == "__main__":
    main()
