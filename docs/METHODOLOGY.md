# Measurement methodology

The canon every skill in this repo cites. Skills reference this file wherever they
say "Verdict rigor", "Full-context reporting", "Grind Mandate", "Asset validation",
or "Keep it tight (no AI-slop)".

## Verdict rigor: DRAFT vs VERDICT

Every performance claim is labeled either **DRAFT** (a single observation, an
extrapolation, or a number whose capture hygiene is unverified) or **VERDICT**
(reproduced, capture-validated, and stated with its full context). A DRAFT must
name what would promote it - the missing rerun, the counter to check, the
control to hold. Never let a DRAFT number travel without its label.

## Full-context reporting

A performance number without its context is noise. Every reported number carries,
inline or in an adjacent table:

- hardware (GPU model, count, interconnect) and topology (TP / PP / EP / DP),
- precision / quantization of weights, activations, and KV cache,
- engine + version (vLLM, SGLang, TensorRT-LLM…) and the launch flags that matter,
- workload shape (input/output lengths, concurrency, dataset),
- what the comparison baseline is, and what changed between the two runs,
- source attribution: the exact command, config file, or bundle the number came from.

Percent claims state their denominator. Speedups state both absolute values.

## Speed-of-light framing

Measured throughput is graded against hardware ceilings, not against vibes.
[`configs/sol-ceilings.yaml`](../configs/sol-ceilings.yaml) is the single source
of truth for published peaks (FLOPS, memory and interconnect bandwidth, per chip
variant, with datasheet citations). Never inline these numbers in skills or
reports - load the YAML and reference by key path. A "%SoL" figure names the
ceiling it is a percentage of.

## Asset validation

Every generated asset is validated before it is reported as done: a rendered PDF
is opened and page-counted, an emitted JSON is parsed and row-counted against its
source, a generated config is round-tripped through its consumer's parser, a
plot's axes are sanity-checked against the raw data. "The command exited 0" is
not validation.

## Kernel-work classification

Before optimizing a kernel, classify it: **K**nown-good (matches roofline
expectation - move on), **R**educible (algorithmic or fusion headroom),
**H**idden (launch/sync overhead, not compute), **P**arallelism-starved
(occupancy / load-balance), or **A**ttribution-error (the profiler is lying -
fix capture hygiene first). Climbing the wrong category wastes the engagement.

## Capture hygiene

An empty or implausible profile is a capture bug until proven otherwise. Validate
captures with [`scripts/nsys-validate-capture.sh`](../scripts/nsys-validate-capture.sh)
before drawing conclusions (cudagraph-aware tracing flags, non-idle windows,
adequate duration). For zymtrace / ClickHouse-backed profilers, ingest lag is
real: gate queries on [`scripts/zymtrace-ingest-wait.sh`](../scripts/zymtrace-ingest-wait.sh)
rather than querying immediately after a run.

## Always be grinding (next-lever framing)

Every result section ends by naming the next lever: the single highest-leverage
follow-up the data points at, with its expected magnitude and cost. A report
that ends "done" is a report that ends the engagement. A report that ends
"NEXT LEVER: …" compounds.

## Value proposition

Every change states what it buys: tokens/s/GPU, $/M-tokens, time-to-train, or
joules/token - before-vs-after, with the workload it was measured on. Work that
cannot state its value proposition is exploration (fine - label it as such).

## De-slop (writing style)

Every human-facing artifact this repo emits (reports, PR bodies, ledger rows,
summaries, commit messages) is written plain. The checklist:

- no em-dashes or en-dashes (the #1 AI-slop tell). Plain punctuation,
- minimal bold. No bold-lead bullets,
- verb-led sentences. No fragments posing as findings,
- numbers over adjectives: "2.1x at C=32" beats "significantly faster",
- no marketing language: blazing, powerful, seamless, robust, leverage,
  cutting-edge, game-changing, world-class, supercharge,
- no AI vocabulary: delve, crucial, comprehensive, nuanced, multifaceted,
  furthermore, moreover, additionally, pivotal, landscape, tapestry,
  underscore, foster, showcase, intricate, vibrant, fundamental, interplay,
- no filler phrases: "here's the thing", "let me break this down", "the bottom
  line", "make no mistake",
- no "it's not X, it's Y" framing. Say the Y,
- cut hedging and redundancy. Inline code stays out of narrow table cells,
- no decorative visuals (infra PRs do not embed charts).

For anything rendered (HTML, slides, dashboards), the same discipline covers
visual slop: no purple/indigo gradients, no symmetric 3-column feature grid, no
centered-everything, no emoji as decoration, no decorative blobs.

A sentence that carries no number, decision, or caveat gets deleted. Skills
cite this section wherever they say "Keep it tight (no AI-slop)". The word and
phrase lists merge this repo's canon with the gstack writing rules
(github.com/garrytan/gstack, MIT) and the visual-slop tells it cites.
