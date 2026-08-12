"""Unit tests for the parts of armtune that don't require an Arm host or a
built llama.cpp: mock data generation, row merging, and winner selection.

Run with: python -m pytest tests/ -v   (or: python -m unittest discover tests)
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from armtune import bench, report


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
            self.assertIn("SYNTHETIC DEMO DATA", paths["markdown"].read_text())


if __name__ == "__main__":
    unittest.main()
