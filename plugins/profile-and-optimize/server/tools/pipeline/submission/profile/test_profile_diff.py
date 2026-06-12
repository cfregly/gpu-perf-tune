#!/usr/bin/env python3
"""Unit tests for tools/pipeline/submission/profile/profile_diff.py.

Locks the parser, delta computation, NVTX keyword filter, formatter,
and the markdown renderer. The harness shells out to ``nsys stats``
in production; tests use the ``--baseline-csv-dir`` /
``--candidate-csv-dir`` path so they do not require ``nsys`` on the
runner.
"""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))

import profile_diff  # noqa: E402


def _write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def _make_nvtxsum(path: Path) -> Path:
    return _write(
        path,
        '"Time (%)","Total Time (ns)","Instances","Avg (ns)","Med (ns)",'
        '"Min (ns)","Max (ns)","StdDev (ns)","Range"\n'
        '50.0,1000000000,10,100000000,100000000,90000000,110000000,1000000,"forward"\n'
        '30.0,600000000,10,60000000,60000000,55000000,65000000,500000,"backward"\n'
        '10.0,200000000,10,20000000,20000000,18000000,22000000,200000,"optimizer"\n'
        '5.0,100000000,5,20000000,20000000,18000000,22000000,200000,"tp_comm_overlap"\n'
        '5.0,100000000,5,20000000,20000000,18000000,22000000,200000,"misc_unrelated"\n',
    )


def _make_gpukernsum(path: Path) -> Path:
    return _write(
        path,
        '"Time (%)","Total Time (ns)","Instances","Avg (ns)","Min (ns)","Max (ns)","Name"\n'
        '50.0,500000000,1000,500000,400000,600000,"gemm_kernel_a"\n'
        '30.0,300000000,1000,300000,250000,350000,"layernorm_kernel"\n'
        '20.0,200000000,500,400000,300000,500000,"mxfp8_cast_kernel"\n',
    )


def _make_cudaapisum(path: Path) -> Path:
    return _write(
        path,
        '"Time (%)","Total Time (ns)","Num Calls","Avg (ns)","Min (ns)",'
        '"Max (ns)","StdDev (ns)","Name"\n'
        '80.0,200000000,5000,40000,30000,50000,1000,"cudaLaunchKernel"\n'
        '20.0,50000000,100,500000,400000,600000,5000,"cudaMemcpyAsync"\n',
    )


def _make_nccltrace(path: Path) -> Path:
    return _write(
        path,
        '"Start (ns)","Duration (ns)","Function","Comm","Stream"\n'
        '0,5000000,"AllReduce","comm0","stream0"\n'
        '6000000,5000000,"AllReduce","comm0","stream0"\n'
        '12000000,3000000,"AllGather","comm0","stream0"\n',
    )


class ParseTests(unittest.TestCase):
    def test_parse_nvtx_extracts_range_total_and_instances(self) -> None:
        with TemporaryDirectory() as tmp:
            csv = _make_nvtxsum(Path(tmp) / "run_a_nvtxsum.csv")
            rows = profile_diff.parse_nvtx(csv)
        names = [r.name for r in rows]
        self.assertIn("forward", names)
        forward = next(r for r in rows if r.name == "forward")
        self.assertEqual(forward.total_ns, 1_000_000_000)
        self.assertEqual(forward.count, 10)

    def test_parse_kernels(self) -> None:
        with TemporaryDirectory() as tmp:
            csv = _make_gpukernsum(Path(tmp) / "run_a_gpukernsum.csv")
            rows = profile_diff.parse_kernels(csv)
        names = {r.name for r in rows}
        self.assertEqual(names, {"gemm_kernel_a", "layernorm_kernel", "mxfp8_cast_kernel"})

    def test_parse_cuda_api_handles_eight_column_header(self) -> None:
        with TemporaryDirectory() as tmp:
            csv = _make_cudaapisum(Path(tmp) / "run_a_cudaapisum.csv")
            rows = profile_diff.parse_cuda_api(csv)
        api_rows = {r.name: r for r in rows}
        self.assertIn("cudaLaunchKernel", api_rows)
        self.assertEqual(api_rows["cudaLaunchKernel"].total_ns, 200_000_000)
        self.assertEqual(api_rows["cudaLaunchKernel"].count, 5000)

    def test_parse_nccl_aggregates_duplicates(self) -> None:
        with TemporaryDirectory() as tmp:
            csv = _make_nccltrace(Path(tmp) / "run_a_nccltrace.csv")
            rows = profile_diff.parse_nccl(csv)
        agg = {r.name: r for r in rows}
        self.assertEqual(agg["AllReduce"].total_ns, 10_000_000)
        self.assertEqual(agg["AllReduce"].count, 2)
        self.assertEqual(agg["AllGather"].total_ns, 3_000_000)
        self.assertEqual(agg["AllGather"].count, 1)

    def test_parse_nvtx_strict_on_malformed(self) -> None:
        """Truncated rows (fewer cells than the header) are silently
        skipped, not silently treated as zeros - this is the strict
        behavior we want so a malformed CSV does not pollute the diff.
        """
        with TemporaryDirectory() as tmp:
            csv = _write(
                Path(tmp) / "x_nvtxsum.csv",
                '"Time (%)","Total Time (ns)","Instances","Avg (ns)","Med (ns)",'
                '"Min (ns)","Max (ns)","StdDev (ns)","Range"\n'
                '50.0,1000000000,10,100000000,100000000,90000000,110000000,1000000,"good"\n'
                '50.0,truncated_row\n',
            )
            rows = profile_diff.parse_nvtx(csv)
        self.assertEqual([r.name for r in rows], ["good"])


class NvtxFilterTests(unittest.TestCase):
    def test_keep_interesting_filters_uninteresting(self) -> None:
        rows = [
            profile_diff.StatRow("forward", 1, 1),
            profile_diff.StatRow("misc_unrelated", 1, 1),
            profile_diff.StatRow("tp_comm_overlap", 1, 1),
            profile_diff.StatRow("alltoall_dispatch", 1, 1),
        ]
        kept = profile_diff.keep_interesting_nvtx(rows)
        names = {r.name for r in kept}
        self.assertEqual(names, {"forward", "tp_comm_overlap", "alltoall_dispatch"})

    def test_keep_interesting_returns_all_when_none_match(self) -> None:
        """Fallback so a malformed run still produces a diff."""
        rows = [profile_diff.StatRow(f"misc_{i}", 1, 1) for i in range(3)]
        kept = profile_diff.keep_interesting_nvtx(rows)
        self.assertEqual(len(kept), 3)


class DeltaTests(unittest.TestCase):
    def test_compute_delta_signs_and_pct(self) -> None:
        baseline = [
            profile_diff.StatRow("a", 1_000, 10),
            profile_diff.StatRow("b", 500, 5),
            profile_diff.StatRow("c", 200, 2),
        ]
        candidate = [
            profile_diff.StatRow("a", 1_500, 10),
            profile_diff.StatRow("b", 250, 5),
            profile_diff.StatRow("d", 100, 1),
        ]
        deltas = profile_diff.compute_delta(baseline, candidate)
        names = {r.name for r in deltas}
        self.assertEqual(names, {"a", "b", "c", "d"})
        deltas_by_name = {r.name: r for r in deltas}
        self.assertEqual(deltas_by_name["a"].delta_ns, 500)
        self.assertAlmostEqual(deltas_by_name["a"].pct, 50.0)
        self.assertEqual(deltas_by_name["b"].delta_ns, -250)
        self.assertAlmostEqual(deltas_by_name["b"].pct, -50.0)
        self.assertEqual(deltas_by_name["c"].delta_ns, -200)
        self.assertEqual(deltas_by_name["c"].candidate_ns, 0)
        self.assertEqual(deltas_by_name["d"].baseline_ns, 0)
        self.assertIsNone(deltas_by_name["d"].pct)

    def test_compute_delta_sorted_by_abs(self) -> None:
        baseline = [profile_diff.StatRow("small", 10, 1), profile_diff.StatRow("big", 1000, 1)]
        candidate = [profile_diff.StatRow("small", 110, 1), profile_diff.StatRow("big", 1500, 1)]
        deltas = profile_diff.compute_delta(baseline, candidate)
        self.assertEqual(deltas[0].name, "big")
        self.assertEqual(deltas[1].name, "small")


class FormatTests(unittest.TestCase):
    def test_fmt_ns_units(self) -> None:
        self.assertEqual(profile_diff.fmt_ns(0), "0")
        self.assertEqual(profile_diff.fmt_ns(123), "123ns")
        self.assertEqual(profile_diff.fmt_ns(12_345), "12.345us")
        self.assertEqual(profile_diff.fmt_ns(12_345_678), "12.346ms")
        self.assertEqual(profile_diff.fmt_ns(12_345_678_901), "12.346s")
        self.assertEqual(profile_diff.fmt_ns(-12_345_678), "-12.346ms")

    def test_fmt_pct(self) -> None:
        self.assertEqual(profile_diff.fmt_pct(None), "n/a")
        self.assertEqual(profile_diff.fmt_pct(50.0), "+50.0%")
        self.assertEqual(profile_diff.fmt_pct(-25.0), "-25.0%")


class RendererTests(unittest.TestCase):
    def test_render_table_handles_empty(self) -> None:
        out = profile_diff.render_table("Empty", [], limit=20)
        self.assertIn("### Empty", out)
        self.assertIn("_No data._", out)

    def test_render_table_includes_columns(self) -> None:
        deltas = [
            profile_diff.DeltaRow(
                name="forward",
                baseline_ns=1_000_000_000,
                candidate_ns=1_500_000_000,
                delta_ns=500_000_000,
                pct=50.0,
                baseline_count=10,
                candidate_count=10,
            ),
        ]
        out = profile_diff.render_table("NVTX", deltas, limit=20)
        self.assertIn("forward", out)
        self.assertIn("1.000s", out)
        self.assertIn("1.500s", out)
        self.assertIn("+50.0%", out)
        self.assertIn("10 -> 10", out)


class EndToEndTests(unittest.TestCase):
    def _write_full_csv_dir(self, root: Path, *, scale: float) -> Path:
        root.mkdir(parents=True, exist_ok=True)
        _write(
            root / "run_nvtxsum.csv",
            '"Time (%)","Total Time (ns)","Instances","Avg (ns)","Med (ns)",'
            '"Min (ns)","Max (ns)","StdDev (ns)","Range"\n'
            f'50.0,{int(1_000_000_000 * scale)},10,0,0,0,0,0,"forward"\n'
            f'30.0,{int(600_000_000 * scale)},10,0,0,0,0,0,"backward"\n'
            f'5.0,{int(100_000_000 * scale)},5,0,0,0,0,0,"tp_comm_overlap"\n',
        )
        _write(
            root / "run_gpukernsum.csv",
            '"Time (%)","Total Time (ns)","Instances","Avg (ns)","Min (ns)","Max (ns)","Name"\n'
            f'50.0,{int(500_000_000 * scale)},1000,0,0,0,"gemm_kernel_a"\n'
            f'30.0,{int(300_000_000 * scale)},1000,0,0,0,"layernorm_kernel"\n',
        )
        _write(
            root / "run_cudaapisum.csv",
            '"Time (%)","Total Time (ns)","Num Calls","Avg (ns)","Min (ns)",'
            '"Max (ns)","StdDev (ns)","Name"\n'
            f'80.0,{int(200_000_000 * scale)},5000,0,0,0,0,"cudaLaunchKernel"\n',
        )
        _write(
            root / "run_nccltrace.csv",
            '"Start (ns)","Duration (ns)","Function","Comm","Stream"\n'
            f'0,{int(5_000_000 * scale)},"AllReduce","comm0","stream0"\n'
            f'10000000,{int(5_000_000 * scale)},"AllReduce","comm0","stream0"\n'
            f'20000000,{int(3_000_000 * scale)},"AllGather","comm0","stream0"\n',
        )
        return root

    def test_main_with_csv_dirs_emits_report_and_json(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            base = self._write_full_csv_dir(root / "baseline", scale=1.0)
            cand = self._write_full_csv_dir(root / "candidate", scale=1.5)
            out_md = root / "out" / "profile-diff.md"
            out_json = root / "out" / "profile-diff.json"
            rc = profile_diff.main(
                [
                    "--baseline-csv-dir", str(base),
                    "--candidate-csv-dir", str(cand),
                    "--baseline-label", "B",
                    "--candidate-label", "C",
                    "--out", str(out_md),
                    "--json-out", str(out_json),
                    "--limit", "5",
                ]
            )
            self.assertEqual(rc, 0)
            report = out_md.read_text(encoding="utf-8")
            self.assertIn("# Profile diff", report)
            self.assertIn("Baseline: `B`", report)
            self.assertIn("Candidate: `C`", report)
            self.assertIn("forward", report)
            self.assertIn("AllReduce", report)
            self.assertIn("gemm_kernel_a", report)
            self.assertIn("cudaLaunchKernel", report)
            sidecar = json.loads(out_json.read_text(encoding="utf-8"))
            self.assertEqual(sidecar["baseline"], "B")
            self.assertEqual(sidecar["candidate"], "C")
            nvtx_names = {row["name"] for row in sidecar["nvtx"]}
            self.assertIn("forward", nvtx_names)
            forward = next(r for r in sidecar["nvtx"] if r["name"] == "forward")
            self.assertEqual(forward["baseline_ns"], 1_000_000_000)
            self.assertEqual(forward["candidate_ns"], 1_500_000_000)
            self.assertEqual(forward["delta_ns"], 500_000_000)
            self.assertAlmostEqual(forward["pct"], 50.0)


class CsvDirCollectorTests(unittest.TestCase):
    def test_collect_csv_dir_complains_on_missing(self) -> None:
        with TemporaryDirectory() as tmp:
            with self.assertRaises(FileNotFoundError):
                profile_diff.collect_csv_dir(Path(tmp))

    def test_collect_csv_dir_complains_on_ambiguous(self) -> None:
        with TemporaryDirectory() as tmp:
            d = Path(tmp)
            for stem in ("run_a", "run_b"):
                _make_nvtxsum(d / f"{stem}_nvtxsum.csv")
                _make_gpukernsum(d / f"{stem}_gpukernsum.csv")
                _make_cudaapisum(d / f"{stem}_cudaapisum.csv")
                _make_nccltrace(d / f"{stem}_nccltrace.csv")
            with self.assertRaises(RuntimeError) as cm:
                profile_diff.collect_csv_dir(d)
            self.assertIn("ambiguous", str(cm.exception))


if __name__ == "__main__":
    unittest.main()
