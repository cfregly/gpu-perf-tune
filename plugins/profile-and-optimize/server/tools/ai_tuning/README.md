# AI-assisted MLPerf tuner

`ai_tuning.py` is the offline parameter-sweep proposal engine for
GB300 + B200 MLPerf Training v6.0 benchmarks. It runs LLM-assisted
Bayesian / TPE / Hyperband / BOHB sweeps over an audited tuning space,
produces config patches that route through the audited safe-template
path, and manages a JSONL experiment ledger. **It is not on the
submission gate.** It is the operator-side helper for selecting the
next set of fabric-knob, NCCL-knob, and config-shape A/Bs to run.

The operator-facing verb and safety contract lives in
[`docs/cli-contract.md`](../../docs/cli-contract.md). The
local promoted candidate ledger lives in
[`tuning/best-known/`](../../tuning/best-known).

## Layout

| Path | Purpose |
| --- | --- |
| [`ai_tuning.py`](ai_tuning.py) | the CLI entry point (single-file front end). |
| [`optimizer/`](optimizer) | optimizer engines: TPE (`tpe.py`), GP-Bayesian (`gp.py`), Hyperband (`hyperband.py`), shared types (`types.py`), tuning-space schema (`space.py`), `.hyp` format I/O (`hyp_format.py`), durable session state (`hyp_session.py`). |
| [`test_ai_tuning.py`](test_ai_tuning.py) | end-to-end CLI tests (subprocess-based). |
| [`test_optimizer.py`](test_optimizer.py) | engine unit tests. |

Pass a tuning-space JSON file with `--space`. The repository does not ship a
production tuning space because valid ranges depend on the workload and
cluster. Bundled example spaces support inspection and offline tests only.
Proposal generation, proposal validation, reports, optimizer proposals, and
experiment creation require an explicit `--space`. Promoted local candidates
can be recorded under [`tuning/best-known/`](../../tuning/best-known).

## CLI surface

Subcommand families (run `python3 tools/ai_tuning/ai_tuning.py --help`
for full options):

| Family | Subcommands | Purpose |
| --- | --- | --- |
| `space` | (top-level) | describe a tuning space (parameter list, ranges). |
| `matrix` | (top-level) | print the proposal-by-parameter matrix view. |
| `optimizer` | `propose`, `status`, `history`, `compare`, `import-hyp` | run the LLM-assisted optimizer. Persist optimizer state across sessions. Ingest hypertune `.hyp` templates. |
| `proposal` | `validate` | validate a proposal JSON against its tuning space. Enforce `requires_config_patch` for any parameter consumed via `config_patches`. |
| `experiment` | `create`, `update`, `summary`, `submit`, `poll`, `collect` | maintain a JSONL ledger of one or more concurrent experiments. Submit (gated), poll Slurm read-only, collect artifacts. |
| `template-patch` | `validate` | validate a context-anchored template patch against the canonical config before submission. |
| `report` | (top-level) | render a tuning campaign summary across a session. |
| `finalize` | (top-level) | mark a campaign as complete and emit the promotion record. |

## Operator gates

The CLI is intentionally read-only by default. Mutating cluster work is
gated by **two** explicit flags:

- `experiment submit` only invokes `sbatch` when **both** `--execute`
  and `--i-understand-this-submits-jobs` are passed in the same
  invocation. Without both, it prints the canonical sbatch line and
  exits 0.
- `experiment poll` is read-only against Slurm (no `scancel`, no
  destructive `scontrol`).
- `experiment collect` copies local artifacts and runs validators only,
  it never invokes mutating cluster commands.
- `template-patch validate` and `proposal validate` never invoke any
  cluster command.

## Local-dev verification

```bash
python3 -m unittest \
  tools/ai_tuning/test_ai_tuning.py \
  tools/ai_tuning/test_optimizer.py
```

Both modules are part of the broader `unittest` battery in the
top-level [`README.md`](../../README.md) "Local dev / verification".

## Cross-references

- [`docs/cli-contract.md`](../../docs/cli-contract.md) -
  the operator contract for arguments, output, and safety labels.
- [`tuning/best-known/`](../../tuning/best-known) -
  the current promoted candidates and their A/B evidence.
