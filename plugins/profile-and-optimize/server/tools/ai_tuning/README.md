# AI-assisted MLPerf tuner

`ai_tuning.py` is the offline parameter-sweep proposal engine for
GB300 + B200 MLPerf Training v6.0 benchmarks. It runs LLM-assisted
Bayesian / TPE / Hyperband / BOHB sweeps over an audited tuning space,
produces config patches that route through the audited safe-template
path, and manages a JSONL experiment ledger. **It is not on the
submission gate.** It is the operator-side helper for selecting the
next set of fabric-knob, NCCL-knob, and config-shape A/Bs to run.

The full operator contract lives in
[`docs/ai-assisted-tuning.md`](/plugins/profile-and-optimize/server/docs/ai-assisted-tuning.md). The
current promoted candidate lineup lives in
[`tuning/best-known/`](/plugins/profile-and-optimize/server/tuning/best-known).

## Layout

| Path | Purpose |
| --- | --- |
| [`ai_tuning.py`](/plugins/profile-and-optimize/server/tools/ai_tuning/ai_tuning.py) | the CLI entry point (single-file front end). |
| [`optimizer/`](/plugins/profile-and-optimize/server/tools/ai_tuning/optimizer) | optimizer engines: TPE (`tpe.py`), GP-Bayesian (`gp.py`), Hyperband (`hyperband.py`), shared types (`types.py`), tuning-space schema (`space.py`), `.hyp` format I/O (`hyp_format.py`), durable session state (`hyp_session.py`). |
| [`test_ai_tuning.py`](/plugins/profile-and-optimize/server/tools/ai_tuning/test_ai_tuning.py) | end-to-end CLI tests (subprocess-based). |
| [`test_optimizer.py`](/plugins/profile-and-optimize/server/tools/ai_tuning/test_optimizer.py) | engine unit tests. |

Tuning spaces are checked in under [`tuning/`](/plugins/profile-and-optimize/server/tuning):

- [`tuning/tuning-space.b200-llama31-8b.json`](/plugins/profile-and-optimize/server/tuning/tuning-space.b200-llama31-8b.json) - B200 LLaMA 3.1 8B operational sweep space.
- [`tuning/tuning-space.gb300-ops.json`](/plugins/profile-and-optimize/server/tuning/tuning-space.gb300-ops.json) - GB300 fabric / NCCL / config knobs sweep space.
- [`tuning/best-known/`](/plugins/profile-and-optimize/server/tuning/best-known) - promoted candidates with A/B evidence. Refreshed after every successful tuning campaign.
- [`tuning/schemas/`](/plugins/profile-and-optimize/server/tuning/schemas) - JSON Schemas for the tuning-space and proposal shapes.

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
top-level [`README.md`](/plugins/profile-and-optimize/server/README.md) "Local dev / verification".

## Cross-references

- [`docs/ai-assisted-tuning.md`](/plugins/profile-and-optimize/server/docs/ai-assisted-tuning.md) -
  the operator contract: parameter audit policy, durable optimizer
  state, MLPerf legality gates, `.hyp` import semantics.
- [`tuning/best-known/`](/plugins/profile-and-optimize/server/tuning/best-known) -
  the current promoted candidates and their A/B evidence.
- [`docs/private-testing.md`](/plugins/profile-and-optimize/server/docs/private-testing.md) - the
  live 8-node sweep workflow this CLI feeds into.
