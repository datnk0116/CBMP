# Artifact Regeneration Provenance

Tables and figures were regenerated strictly from existing committed JSONL experiment logs (without re-running multi-hour SAT/MIP/CP experiments) after applying audit fixes **A1–A6** and dataset verification **B1–B3** on branch `fix/audit3`.

## Code Version
- Analysis & encoding base commit: `6dbcc9c` (`fix/audit3`, incorporating A1–A6 fixes, B1–B3 dataset manifest/config updates, and `get_lb_degree` symmetrization)
- Generator scripts:
  - `analysis/make_tables.py`
  - `analysis/make_plots.py`

## Source JSONL Inputs & Recorded Run Commits
1. `docs/data/runs.jsonl` (207 records)
   - Embedded `git_commit`s: `499d4c94e846875ef5de4645bc0a41f65f2e38fc` (142), `d04ca7edc719af0b57d47f8267fb960feddf4e1c` (42), `3aa3bd610c2e842f6ceae7e9f7658694e646a058` (16), `96b1bde2508510a879da199d44320f74c5073627` (7)
   - Regenerated targets: `docs/phaseB/`, `docs/phaseC/`, `report/figures/cactus_{solvers_small,solvers_medium,methods}.{png,pdf}`
2. `docs/data/runs_ablation.jsonl` (175 aggregated / 255 raw lines)
   - Embedded `git_commit`s: `89603149159ef65b2b6a5220ad59a80d80a0d09c` (175)
   - Regenerated targets: `docs/phaseA/`, `docs/phaseA2/`, `docs/phaseD/`, `report/figures/*ablation*`, `report/figures/scatter_sb_vs_nosb_small.*`, `report/figures/boxplot_ablation_family.*`
3. `results_v2/runs.jsonl` (227 aggregated / 443 raw lines; note `results_v2/` outputs are untracked local runtime artifacts)
   - Embedded `git_commit`s: `499d4c94e846875ef5de4645bc0a41f65f2e38fc` (138), `d04ca7edc719af0b57d47f8267fb960feddf4e1c` (42), `3e5604b6cb79888c9c8663a40408b369049e29d4` (24), `3aa3bd610c2e842f6ceae7e9f7658694e646a058` (16), `96b1bde2508510a879da199d44320f74c5073627` (7)
   - Regenerated targets: `results_v2/tables/`, `results_v2/figures/`

## Notes
- Previous artifacts in `docs/phase*` and `report/figures/` had been generated prior to the A1–A4 analysis fixes (and with older code revisions matching the recorded JSONL commits).
- Re-running the full benchmark suite on current solver code (including `harwell_boeing_28`) requires explicit execution of `experiments/run_experiments.py` and was not performed automatically.

