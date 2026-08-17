"""Unit tests for the parts of armtune that don't require an Arm host or a
built llama.cpp: mock data generation, row merging, and winner selection.

Run with: python -m pytest tests/ -v   (or: python -m unittest discover tests)
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from armtune import bench, htmlreport, quantize, report, serve
from armtune.livestate import SweepState
from armtune.liveserver import LiveServer


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
            for key in ("raw_json", "csv", "markdown", "html", "launch_script"):
                self.assertTrue(paths[key].exists(), f"missing artifact: {key}")
            md = paths["markdown"].read_text()
            self.assertIn("SYNTHETIC DEMO DATA", md)
            self.assertIn("Baseline vs. tuned", md)
            self.assertNotIn("$/1M tok", md)  # no cost_per_hour given

            winner = winners["fastest_throughput"]
            launch = paths["launch_script"].read_text()
            self.assertIn(winner.model_path, launch)
            self.assertNotIn("<your-model>", launch)

            html = paths["html"].read_text()
            self.assertIn("<svg", html)
            self.assertIn("SYNTHETIC DEMO DATA", html)

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


class TestHtmlReport(unittest.TestCase):
    def setUp(self):
        self.rows = bench.generate_mock_results(
            quant_types=["Q4_0", "Q8_0"], threads=[2, 4], batch_sizes=[512],
            n_prompt=512, n_gen=128,
        )
        self.results = report.merge_rows(self.rows)
        self.winners = report.pick_winners(self.results)

    def test_render_is_valid_html_with_no_nan_or_infinity(self):
        html = htmlreport.render_html_report(self.results, self.winners, "test-model", "test-cpu")
        self.assertIn("<html", html)
        self.assertIn("</html>", html)
        self.assertNotIn("NaN", html)
        self.assertNotIn("Infinity", html)

    def test_render_includes_one_line_per_quant(self):
        html = htmlreport.render_html_report(self.results, self.winners, "test-model", "test-cpu")
        self.assertIn(">Q4_0<", html)
        self.assertIn(">Q8_0<", html)

    def test_render_includes_cost_chart_only_when_given(self):
        without_cost = htmlreport.render_html_report(self.results, self.winners, "test-model", "test-cpu")
        with_cost = htmlreport.render_html_report(
            self.results, self.winners, "test-model", "test-cpu", cost_per_hour=0.0672,
        )
        self.assertNotIn("$ per 1M generated tokens", without_cost)
        self.assertIn("$ per 1M generated tokens", with_cost)

    def test_render_includes_concurrency_section_only_when_given(self):
        without = htmlreport.render_html_report(self.results, self.winners, "test-model", "test-cpu")
        conc = serve.generate_mock_concurrency([1, 4], single_stream_tok_s=40.0)
        with_conc = htmlreport.render_html_report(
            self.results, self.winners, "test-model", "test-cpu", concurrency_results=conc,
        )
        self.assertNotIn("Concurrent serving throughput", without)
        self.assertIn("Concurrent serving throughput", with_conc)

    def test_bar_chart_handles_empty_input(self):
        self.assertIn("No data", htmlreport._svg_bar_chart([], title="t", y_label="y"))

    def test_line_chart_handles_single_point_without_division_by_zero(self):
        svg = htmlreport._svg_line_chart({"Q4_0": [(4, 30.0)]}, title="t", x_label="x", y_label="y")
        self.assertIn("<svg", svg)
        self.assertNotIn("NaN", svg)


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


class TestSweepState(unittest.TestCase):
    def test_set_status_updates_status_and_message(self):
        s = SweepState()
        s.set_status("quantizing", "Quantizing Q4_0...")
        d = s.to_dict()
        self.assertEqual(d["status"], "quantizing")
        self.assertEqual(d["message"], "Quantizing Q4_0...")

    def test_advance_increments_steps_done(self):
        s = SweepState()
        s.set_total_steps(4)
        s.advance()
        s.advance("halfway")
        d = s.to_dict()
        self.assertEqual(d["steps_done"], 2)
        self.assertEqual(d["total_steps"], 4)
        self.assertEqual(d["message"], "halfway")

    def test_progress_lines_capped(self):
        s = SweepState()
        for i in range(100):
            s.add_progress_line(f"line {i}")
        d = s.to_dict()
        self.assertEqual(len(d["progress_lines"]), 60)
        self.assertEqual(d["progress_lines"][-1], "line 99")

    def test_set_error_sets_status_error(self):
        s = SweepState()
        s.set_error("something broke")
        d = s.to_dict()
        self.assertEqual(d["status"], "error")
        self.assertEqual(d["error"], "something broke")

    def test_to_dict_has_expected_keys(self):
        d = SweepState().to_dict()
        for key in ("status", "message", "progress_lines", "total_steps", "steps_done", "error", "updated_at"):
            self.assertIn(key, d)


class TestLiveServer(unittest.TestCase):
    def test_serves_state_report_and_404s_missing_files(self):
        import json
        import tempfile
        import urllib.error
        import urllib.request

        state = SweepState()
        state.set_total_steps(3)
        state.advance("Quantizing Q4_0...")

        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp)
            server = LiveServer(state, out_dir=out_dir, port=0)
            server.start()
            try:
                base = server.url

                with urllib.request.urlopen(base) as r:
                    self.assertEqual(r.status, 200)
                    self.assertIn(b"<html", r.read())

                with urllib.request.urlopen(base + "state.json") as r:
                    body = json.loads(r.read())
                    self.assertEqual(body["steps_done"], 1)
                    self.assertEqual(body["total_steps"], 3)

                # report.html doesn't exist yet -> 404
                try:
                    urllib.request.urlopen(base + "report.html")
                    self.fail("expected HTTPError for missing report.html")
                except urllib.error.HTTPError as e:
                    self.assertEqual(e.code, 404)

                # write it, now it should serve
                (out_dir / "report.html").write_text("<html>final report</html>")
                with urllib.request.urlopen(base + "report.html") as r:
                    self.assertEqual(r.status, 200)
                    self.assertIn(b"final report", r.read())
            finally:
                server.stop()


class TestQuantizeCallbacks(unittest.TestCase):
    def _fake_binary(self, tmp: Path, exit_code: int = 0) -> Path:
        script = tmp / "fake-llama-quantize"
        script.write_text(
            "#!/bin/sh\n"
            f"touch \"$2\"\n"
            f"exit {exit_code}\n"
        )
        script.chmod(0o755)
        return script

    def test_start_and_done_callbacks_fire_per_quant_in_order(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            base_model = tmp_path / "model.gguf"
            base_model.write_text("fake")
            quantize_bin = self._fake_binary(tmp_path)
            cache_dir = tmp_path / "models"

            events = []
            out = quantize.ensure_quantized(
                base_model, ["Q4_0", "Q8_0"], cache_dir, quantize_bin,
                on_quant_start=lambda q: events.append(("start", q)),
                on_quant_done=lambda q: events.append(("done", q)),
            )
            self.assertEqual(events, [("start", "Q4_0"), ("done", "Q4_0"), ("start", "Q8_0"), ("done", "Q8_0")])
            self.assertEqual(set(out.keys()), {"Q4_0", "Q8_0"})

    def test_failure_raises_and_skips_done_callback(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            base_model = tmp_path / "model.gguf"
            base_model.write_text("fake")
            quantize_bin = self._fake_binary(tmp_path, exit_code=1)
            cache_dir = tmp_path / "models"

            events = []
            with self.assertRaises(quantize.QuantizeError):
                quantize.ensure_quantized(
                    base_model, ["Q4_0"], cache_dir, quantize_bin,
                    on_quant_start=lambda q: events.append(("start", q)),
                    on_quant_done=lambda q: events.append(("done", q)),
                )
            self.assertEqual(events, [("start", "Q4_0")])


class TestBenchCallbacksAndStreaming(unittest.TestCase):
    def _fake_bench_binary(self, tmp: Path, exit_code: int = 0) -> Path:
        script = tmp / "fake-llama-bench"
        script.write_text(
            "#!/usr/bin/env python3\n"
            "import sys, json\n"
            "print('progress line 1', file=sys.stderr)\n"
            "print('progress line 2', file=sys.stderr)\n"
            "rows = [{'n_prompt': 512, 'n_gen': 0, 'n_threads': 4, 'n_batch': 512,\n"
            "         'model_filename': 'm.gguf', 'model_size': 100, 'model_n_params': 1,\n"
            "         'avg_ns': 1000000000, 'stddev_ns': 0, 'avg_ts': 512.0, 'stddev_ts': 0},\n"
            "        {'n_prompt': 0, 'n_gen': 128, 'n_threads': 4, 'n_batch': 512,\n"
            "         'model_filename': 'm.gguf', 'model_size': 100, 'model_n_params': 1,\n"
            "         'avg_ns': 1000000000, 'stddev_ns': 0, 'avg_ts': 30.0, 'stddev_ts': 0}]\n"
            "print(json.dumps(rows))\n"
            f"sys.exit({exit_code})\n"
        )
        script.chmod(0o755)
        return script

    def test_streams_progress_and_tags_rows_with_quant(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            bench_bin = self._fake_bench_binary(tmp_path)

            progress_lines = []
            quant_events = []
            rows = bench.run_llama_bench(
                bench_bin=bench_bin, model_paths={"Q4_0": tmp_path / "m.gguf"},
                threads=[4], batch_sizes=[512], n_prompt=512, n_gen=128,
                on_quant_start=lambda q: quant_events.append(("start", q)),
                on_quant_done=lambda q: quant_events.append(("done", q)),
                on_progress_line=progress_lines.append,
            )
            self.assertEqual(len(rows), 2)
            self.assertTrue(all(r["armtune_quant"] == "Q4_0" for r in rows))
            self.assertEqual(quant_events, [("start", "Q4_0"), ("done", "Q4_0")])
            self.assertEqual(progress_lines, ["progress line 1", "progress line 2"])

    def test_nonzero_exit_raises_bench_error(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            bench_bin = self._fake_bench_binary(tmp_path, exit_code=1)
            with self.assertRaises(bench.BenchError):
                bench.run_llama_bench(
                    bench_bin=bench_bin, model_paths={"Q4_0": tmp_path / "m.gguf"},
                    threads=[4], batch_sizes=[512], n_prompt=512, n_gen=128,
                )


if __name__ == "__main__":
    unittest.main()
