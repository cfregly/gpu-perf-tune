# Synthetic performance-hint estimate

This example shows the artifact shape produced by `inference-performance-hints`.
It is intentionally synthetic. None of the rates or latencies describe a real
GPU, model deployment, or measured campaign.

The scenario is a 70B dense model with 8-bit weights at TP8. The estimate assumes
an achieved 2.0 TB/s of HBM bandwidth per GPU, not a datasheet peak. Streaming
70 GB of weights across 16 TB/s of aggregate achieved bandwidth gives a 4.375 ms
weight-traffic lower bound per decode step. The launch and collective estimates
are smaller, so the overlap-aware lower bound is 4.375 ms, not the sum of all
three terms.

A synthetic 6.0 ms measured step is 72.9% of that estimated ceiling. The 1.625 ms
gap ranks the next investigation. It does not prove the gap is caused by launches.
The next experiment captures a clean c=1 decode-step budget and stops if host gaps
are below 5% of step time.

Parse the fixture with:

```bash
python3 -m json.tool examples/performance-hints/decode-estimate.json >/dev/null
```

The example keeps every estimate labeled DRAFT and records source and confidence
for each assumed cost.
