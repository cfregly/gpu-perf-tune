# Third-Party Notices and Provenance

This file records third-party material and known provenance gaps in this
repository. The root MIT license applies to project-authored material. It does
not replace a third party's license or attribution requirements.

## Contributor Covenant

`CODE_OF_CONDUCT.md` contains Contributor Covenant version 2.1 with a project
reporting address. The upstream text is licensed under the Creative Commons
Attribution 4.0 International license. The local copy changes the reporting
method and project contact while retaining the upstream attribution and links.

- Upstream: <https://www.contributor-covenant.org/version/2/1/code_of_conduct/>
- License: <https://creativecommons.org/licenses/by/4.0/>
- Local file: `CODE_OF_CONDUCT.md`

## Performance Hints adaptation

The repository adapts *Performance Hints* by Jeff Dean and Sanjay Ghemawat from
`fast/hints.md` in `abseil/abseil.github.io` at commit
[`fe785f6c18a47415c82f3e15511c6dbf7739cc2e`](https://github.com/abseil/abseil.github.io/commit/fe785f6c18a47415c82f3e15511c6dbf7739cc2e).
That source is licensed under the Apache License 2.0. A copy of the license is
included at `LICENSES/Apache-2.0.txt`.
The upstream repository has no `NOTICE` file. Its `LICENSE` file also records
`Copyright 2017 GitHub, Inc.` This notice is retained here.

The local document and skill change the examples, formulas, workflow, and
recommendations for this project's GPU inference work.

- Upstream: <https://abseil.io/fast/hints.html>
- Source: <https://github.com/abseil/abseil.github.io/blob/fe785f6c18a47415c82f3e15511c6dbf7739cc2e/fast/hints.md>
- Primary adaptation: `plugins/profile-and-optimize/server/docs/performance-hints.md`
- Related skill: `plugins/profile-and-optimize/skills/inference-performance-hints/SKILL.md`

## zymtrace skill adaptation

`plugins/profile-and-optimize/skills/analyze-zymtrace-workload/SKILL.md` is
adapted from `skills/analyze-zymtrace-workload/SKILL.md` in
[`zystem-io/zymtrace-skills`](https://github.com/zystem-io/zymtrace-skills) at
commit
[`8ed647df2a9cf18bc4e0249a4d9ceb0552df5ac6`](https://github.com/zystem-io/zymtrace-skills/commit/8ed647df2a9cf18bc4e0249a4d9ceb0552df5ac6).
Israel Ogbole authored that commit on 2026-05-19. The source file identifies
its author as `zymtrace` and its repository as `zystem-io/zymtrace-skills`.

The source revision is licensed under the Apache License 2.0 and does not
contain a `NOTICE` file. A copy of the license is included at
`LICENSES/Apache-2.0.txt`.

The local file has been modified. Changes include local frontmatter, tool
permissions, links to this repository, reporting requirements, and operator
guidance. Synchronization with the upstream project is manual.

## MLPerf-oriented project code audit

A legacy comment in `.pre-commit-config.yaml` described several server paths as
vendored from an unspecified MLPerf Training mirror. Most paths named by that
comment never existed in this repository. The comment did not identify a URL,
source revision, copyright notice, or path-level license.

The historical bundled server ownership statement in
[`plugins/profile-and-optimize/server/AGENTS.md` at commit `65df322`](https://github.com/cfregly/gpu-perf-tune/blob/65df322f7e5d5f1f0075cc1ea53eb0e054fcef50/plugins/profile-and-optimize/server/AGENTS.md)
records the server as project-owned code with no external upstream. That record
supports the current client-neutral policy in
`plugins/profile-and-optimize/server/AGENTS.md`.

A provenance audit on 2026-08-16 compared the initial repository blobs against
the full public Git history of `mlcommons/training`. It found no substantive
match. Public GitHub code searches for distinctive server strings also found
only this repository. Negative searches cannot rule out every private or
deleted source, but they found no evidence that the current server code was
copied from the public MLCommons repository.

Based on the project ownership record and the available evidence, the current
MLPerf-oriented server code is treated as project-authored material under the
root MIT license. MLPerf names describe the benchmark domain. No MLCommons
license terms are attached without an identified source.

Send corrections or additional provenance privately through the
[maintainer contact in `SECURITY.md`](SECURITY.md#report-a-vulnerability-privately).

## External runtime components

Kernel profiling can launch the `ghcr.io/cfregly/nsys-sidecar` container in a
target pod. The image is not stored in this repository. Every checked-in
default and example pins release `0.1.0` to OCI digest
`sha256:3146de96f6022a8cc36f86d1b8c0281cb940e51e2c3dc49c315646ad66ede43d`.
The image labels identify its licenses as `Apache-2.0 AND NVIDIA-EULA` and its
maintainer as NVIDIA Corporation. Operators must review those terms before
running the kernel profiling command. The image labels do not identify a
public source repository.

## Optional external MCP servers

These servers are not bundled or launched by this repository. Operators may
configure them in their own client settings. The table records verified image
digests for operators who choose a container-based installation.

| Component | Pinned release | Upstream source | Upstream license |
| --- | --- | --- | --- |
| Grafana MCP | `1.1.0`, OCI digest `sha256:f21a19cebbfa7c3a76ef1746171e5ffc3601064e432f593e7c6cb526e5216e5f` | <https://github.com/grafana/mcp-grafana> | Apache License 2.0 |
| GitHub MCP Server | `v1.9.0`, OCI digest `sha256:881b53d6f75f69bdbc1b5b10fc2f1361717c19054143b3a8529fb5c32061a50e` | <https://github.com/github/github-mcp-server> | MIT License |

Python packages declared in the two server `pyproject.toml` files are also
resolved from external sources at install time. Their own licenses and notices
continue to apply.

The optional development dependency `skills-ref` is fetched from
<https://github.com/agentskills/agentskills> at commit
`69ef37e9424c0a7ea9dd2293b559e43ec8176379`. That upstream repository uses the
Apache License 2.0. The dependency is not stored in this repository.
