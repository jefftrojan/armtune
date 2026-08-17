"""Unit tests for the parts of armtune that don't require an Arm host or a
built llama.cpp: mock data generation, row merging, and winner selection.

Run with: python -m pytest tests/ -v   (or: python -m unittest discover tests)
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from armtune import bench, report, serve


class TestMockGeneration(unittest.TestCase):
    def test_generates_paired_pp_and_tg_rows(self):
        rows = bench.generate_mock_results(
            quant_types=["Q4_0", "Q8_0"],
            threads=[4, 8],
            batch_sizes=[512],
            n_prompt=512,
            n_gen=128,
        )
        # 2 quants * 2 threads * 1 batch * 2 rows (pp + tg) each
        self.assertEqual(len(rows), 8)
        pp_rows = [r for r in rows if r["n_gen"] == 0]
        tg_rows = [r for r in rows if r["n_prompt"] == 0]
        self.assertEqual(len(pp_rows), 4)
        self.assertEqual(len(tg_rows), 4)

    def test_larger_quant_has_larger_model_size(self):
        rows = bench.generate_mock_results(
            quant_types=["Q4_0", "Q8_0"], threads=[4], batch_sizes=[512],
            n_prompt=512, n_gen=128,
        )
        sizes = {r["armtune_quant"]: r["model_size"] for r in rows}
        self.assertGreater(sizes["Q8_0"], sizes["Q4_0"])


class TestMergeAndRank(unittest.TestCase):
    def setUp(self):
        self.rows = bench.generate_mock_results(
            quant_types=["Q4_0", "Q4_K_M", "Q8_0"],
            threads=[2, 4, 8],
            batch_sizes=[512, 2048],
            n_prompt=512,
            n_gen=128,
        )

    def test_merge_row_count(self):
        results = report.merge_rows(self.rows)
        # 3 quants * 3 thread counts * 2 batch sizes = 18 complete configs
        self.assertEqual(len(results), 18)

    def test_results_sorted_by_throughput_desc(self):
        results = report.merge_rows(self.rows)
        throughputs = [r.tg_tokens_per_s for r in results]
        self.assertEqual(throughputs, sorted(throughputs, reverse=True))

    def test_winners_present_and_consistent(self):
        results = report.merge_rows(self.rows)
        winners = report.pick_winners(results)
        self.assertIn("fastest_throughput", winners)
        self.assertIn("lowest_ttft", winners)
        self.assertIn("best_value", winners)

        fastest = winners["fastest_throughput"]
        self.assertEqual(fastest.tg_tokens_per_s, max(r.tg_tokens_per_s for r in results))

        lowest_ttft = winners["lowest_ttft"]
        self.assertEqual(lowest_ttft.ttft_ms, min(r.ttft_ms for r in results))

        # best_value must be within 10% of fastest and no larger than fastest itself
        best_value = winners["best_value"]
        self.assertGreaterEqual(best_value.tg_tokens_per_s, fastest.tg_tokens_per_s * 0.90)

    def test_best_per_quant_covers_every_quant(self):
        results = report.merge_rows(self.rows)
        winners = report.pick_winners(results)
        for quant in ["Q4_0", "Q4_K_M", "Q8_0"]:
            self.assertIn(f"best_{quant}", winners)

    def test_model_path_propagated_from_bench_rows(self):
        results = report.merge_rows(self.rows)
        for r in results:
            self.assertEqual(r.model_path, f"mock/{r.quant}.gguf")

    def test_baseline_is_least_compressed_quant_at_max_threads_and_batch(self):
        results = report.merge_rows(self.rows)
        baseline = report.pick_baseline(results)
        # Q8_0 has the largest on-disk size in _QUANT_PROFILE, and the sweep
        # covers threads=[2,4,8] / batch=[512,2048], so baseline should be
        # Q8_0 @ 8 threads, batch 2048.
        self.assertEqual(baseline.quant, "Q8_0")
        self.assertEqual(baseline.threads, 8)
        self.assertEqual(baseline.batch, 2048)

    def test_winners_include_baseline(self):
        results = report.merge_rows(self.rows)
        winners = report.pick_winners(results)
        self.assertIn("baseline", winners)
        fastest = winners["fastest_throughput"]
        baseline = winners["baseline"]
        # The tuned winner should never be slower than the untuned baseline.
        self.assertGreaterEqual(fastest.tg_tokens_per_s, baseline.tg_tokens_per_s)

    def test_write_artifacts_creates_expected_files(self):
        import tempfile
        results = report.merge_rows(self.rows)
        winners = report.pick_winners(results)
        with tempfile.TemporaryDirectory() as tmp:
            paths = report.write_artifacts(
                self.rows, results, winners, Path(tmp), "test-model", "test-cpu"
            )
            for key in ("raw_json", "csv", "markdown", "launch_script"):
                self.assertTrue(paths[key].exists(), f"missing artifact: {key}")
            md = paths["markdown"].read_text()
            self.assertIn("SYNTHETIC DEMO DATA", md)
            self.assertIn("Baseline vs. tuned", md)
            self.assertNotIn("$/1M tok", md)  # no cost_per_hour given

            winner = winners["fastest_throughput"]
            launch = paths["launch_script"].read_text()
            self.assertIn(winner.model_path, launch)
            self.assertNotIn("<your-model>", launch)

    def test_write_artifacts_includes_cost_when_given(self):
        import tempfile
        results = report.merge_rows(self.rows)
        winners = report.pick_winners(results)
        with tempfile.TemporaryDirectory() as tmp:
            paths = report.write_artifacts(
                self.rows, results, winners, Path(tmp), "test-model", "test-cpu",
                cost_per_hour=0.0672,
            )
            md = paths["markdown"].read_text()
            self.assertIn("$/1M tok", md)
            self.assertIn("cheaper", md)


class TestMockConcurrency(unittest.TestCase):
    def test_aggregate_throughput_increases_with_concurrency(self):
        results = serve.generate_mock_concurrency([1, 4, 8, 16], single_stream_tok_s=50.0)
        agg = [r["aggregate_tok_s"] for r in results]
        self.assertEqual(agg, sorted(agg))
        self.assertEqual(agg[0], 50.0)  # concurrency=1 matches single-stream

    def test_per_request_throughput_decreases_with_concurrency(self):
        results = serve.generate_mock_concurrency([1, 4, 8, 16], single_stream_tok_s=50.0)
        per_req = [r["per_request_tok_s"] for r in results]
        self.assertEqual(per_req, sorted(per_req, reverse=True))

    def test_flagged_as_mock(self):
        results = serve.generate_mock_concurrency([1, 2], single_stream_tok_s=50.0)
        self.assertTrue(all(r["armtune_mock"] for r in results))


class TestConcurrencyReport(unittest.TestCase):
    def test_render_includes_all_levels(self):
        results = serve.generate_mock_concurrency([1, 4, 8], single_stream_tok_s=50.0)
        md = report.render_concurrency_section(results)
        self.assertIn("Concurrent serving throughput", md)
        for r in results:
            self.assertIn(str(r["concurrency"]), md)

    def test_render_surfaces_errors(self):
        md = report.render_concurrency_section([{"concurrency": 8, "error": "connection refused"}])
        self.assertIn("connection refused", md)

    def test_append_concurrency_writes_raw_json_and_appends_markdown(self):
        import tempfile
        results = serve.generate_mock_concurrency([1, 4], single_stream_tok_s=50.0)
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp)
            md_path = out_dir / "report.md"
            md_path.write_text("# ArmTune sweep report\n")
            raw_path = report.append_concurrency(out_dir, results)
            self.assertTrue(raw_path.exists())
            self.assertIn("Concurrent serving throughput", md_path.read_text())
            self.assertIn("# ArmTune sweep report", md_path.read_text())  # original content preserved


class TestCostPerToken(unittest.TestCase):
    def test_cost_scales_inversely_with_throughput(self):
        slow = report.cost_per_1m_tokens(tg_tokens_per_s=10.0, cost_per_hour=1.0)
        fast = report.cost_per_1m_tokens(tg_tokens_per_s=20.0, cost_per_hour=1.0)
        self.assertAlmostEqual(fast, slow / 2)

    def test_cost_is_dollars_per_hour_over_tokens_per_hour(self):
        # 100 tok/s * 3600 s/hr = 360,000 tok/hr; $1/hr -> $1 per 360k tokens
        # -> $2.7778 per 1M tokens.
        cost = report.cost_per_1m_tokens(tg_tokens_per_s=100.0, cost_per_hour=1.0)
        self.assertAlmostEqual(cost, 1_000_000 / (100.0 * 3600), places=6)


if __name__ == "__main__":
    unittest.main()
