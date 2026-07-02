#!/usr/bin/env python3
"""Smoke tests for the profiling analysis layer.

Stdlib-only, GPU-free, headless: fabricates tiny derived CSVs in a temp results
dir and runs every analysis module end-to-end, then checks the invariants that
matter (unit normalization, stage mapping, verdicts, report generation).

Run:  python3 profiling/tests/test_analysis.py
"""
import json
import os
import shutil
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis import bandwidth, build_dag, common, roofline, screen, stages  # noqa: E402


def make_ncu_csv(path):
    """A 2-launch ncu --csv --page raw export with the characterize metrics."""
    names = ["ID", "Process ID", "Process Name", "Host Name", "Kernel Name",
             "Context", "Stream", "Block Size", "Grid Size", "Device", "CC",
             "gpu__time_duration.sum",
             "gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed",
             "sm__throughput.avg.pct_of_peak_sustained_elapsed",
             "dram__bytes_read.sum", "dram__bytes_write.sum",
             "dram__throughput.avg.pct_of_peak_sustained_elapsed",
             "lts__t_sector_hit_rate.pct", "l1tex__t_sector_hit_rate.pct",
             "sm__warps_active.avg.pct_of_peak_sustained_active",
             "smsp__sass_thread_inst_executed_op_fadd_pred_on.sum",
             "smsp__sass_thread_inst_executed_op_fmul_pred_on.sum",
             "smsp__sass_thread_inst_executed_op_ffma_pred_on.sum",
             "l1tex__t_bytes.sum", "lts__t_bytes.sum",
             "smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio",
             "l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_ld.ratio"]
    units = [""] * 11 + ["us", "%", "%", "Kbyte", "Kbyte", "%", "%", "%", "%",
                         "inst", "inst", "inst", "Mbyte", "Kbyte", "", ""]
    rows = [
        # memory-bound streaming kernel: 100 us, 2 MB DRAM read (as 2,000 Kbyte)
        ["0", "1", "py", "h", "cuvslam::cuda::cast_image_kernel(const unsigned char *)",
         "1", "7", "(256,1,1)", "(32,1,1)", "0", "7.5",
         "100.000000", "85.0", "10.0", '"2,000"', '"500"', "80.0", "60.0", "5.0",
         "88.0", '"1,000,000"', '"500,000"', '"2,000,000"', "12", '"9,000"',
         "25.5", "4.1"],
        # compute-leaning BA kernel
        ["1", "1", "py", "h", "cuvslam::cuda::sba::calc_jacobians_kernel(meta)",
         "1", "7", "(128,1,1)", "(64,1,1)", "0", "7.5",
         "200.000000", "25.0", "70.0", '"100"', '"50"', "20.0", "90.0", "80.0",
         "60.0", '"50,000,000"', '"20,000,000"', '"90,000,000"', "40", '"30,000"',
         "2.0", "4.0"],
    ]
    with open(path, "w") as fh:
        fh.write(",".join(f'"{n}"' for n in names) + "\n")
        fh.write(",".join(f'"{u}"' for u in units) + "\n")
        for r in rows:
            fh.write(",".join(c if c.startswith('"') else f'"{c}"' for c in r) + "\n")


def make_kern_sum(path):
    with open(path, "w") as fh:
        fh.write("Time (%),Total Time (ns),Instances,Avg (ns),Med (ns),Min (ns),"
                 "Max (ns),StdDev (ns),Name\n")
        fh.write('60.0,6000000,300,20000.0,19000.0,1000,90000,100.0,'
                 '"cuvslam::cuda::sba::build_full_system_1_kernel(meta)"\n')
        fh.write('30.0,3000000,120,25000.0,24000.0,2000,80000,200.0,'
                 '"cuvslam::cuda::cast_image_kernel(const unsigned char *)"\n')
        fh.write('10.0,1000000,60,16666.0,16000.0,3000,70000,300.0,'
                 '"void kernel<getrf_wo_pivot_params_<float, (int)0>>(int, void *)"\n')


class TestCommon(unittest.TestCase):
    def test_to_si(self):
        self.assertAlmostEqual(common.to_si("100.5", "us"), 100.5e-6)
        self.assertAlmostEqual(common.to_si("2,000", "Kbyte"), 2e6)
        self.assertAlmostEqual(common.to_si("3", "Gbyte"), 3e9)
        self.assertAlmostEqual(common.to_si("42.5", "%"), 42.5)
        self.assertTrue(common.to_si("", "%") != common.to_si("", "%"))  # NaN

    def test_base_kernel_name(self):
        cases = {
            "cuvslam::cuda::sba::build_full_system_1_kernel(a, b)": "sba::build_full_system_1_kernel",
            "void cub::CUB_300001_SM_750::detail::merge_sort::DeviceMergeSortMergeKernel<X>(a)": "cub::DeviceMergeSortMergeKernel",
            "void kernel<getrf_wo_pivot_params_<float, (int)0>>(int)": "getrf_wo_pivot_params",
            "cuvslam::cuda::lk_track_kernel(P, P)": "lk_track_kernel",
        }
        for full, want in cases.items():
            self.assertEqual(common.base_kernel_name(full), want)

    def test_stage_mapping(self):
        self.assertEqual(stages.stage_of("cast_image_kernel"), "preprocess")
        self.assertEqual(stages.stage_of("sba::calc_jacobians_kernel"), "bundle_adjust")
        self.assertEqual(stages.stage_of("getrf_wo_pivot_params"), "ba_solver")
        self.assertEqual(stages.stage_of("cub::DeviceMergeSortMergeKernel"), "keypoint_sort")
        self.assertEqual(stages.stage_of("loop_closure_match_kernel"), "slam_loop")
        self.assertEqual(stages.stage_of("mystery_kernel_xyz"), "other")


class TestPipeline(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.run_ncu = os.path.join(self.tmp, "2026-01-01_000000_test_ncu_hw")
        self.run_nsys = os.path.join(self.tmp, "2026-01-01_000001_test_nsys_hw")
        for d in (self.run_ncu, self.run_nsys):
            os.makedirs(os.path.join(d, "derived"))
            json.dump({"gpu": {"name": "TestGPU"}},
                      open(os.path.join(d, "metadata.json"), "w"))
        make_ncu_csv(os.path.join(self.run_ncu, "derived", "ncu_metrics.csv"))
        make_kern_sum(os.path.join(self.run_nsys, "derived",
                                   "kern_sum_cuda_gpu_kern_sum.csv"))
        with open(os.path.join(self.run_nsys, "derived", "used_config.toml"), "w") as fh:
            fh.write("[run]\nmax_frames  = 10\n")
        self.hw = {"device": {"name": "TestGPU", "role": "prototype"},
                   "memory": {"dram_gbps_theoretical": 56.0, "l2_bytes": 524288},
                   "compute": {"fp32_tflops_theoretical": 3.76}}

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def test_build_dag(self):
        dag = build_dag.build(self.run_nsys)
        self.assertEqual(dag["frames"], 10)
        self.assertAlmostEqual(dag["kernels_per_frame"], 48.0)
        st = dag["per_stage"]
        self.assertEqual(st["bundle_adjust"]["instances"], 300)
        self.assertEqual(st["ba_solver"]["instances"], 60)   # getrf via inner name
        files = build_dag.emit(dag, os.path.join(self.tmp, "out"))
        self.assertTrue(all(os.path.isfile(f) for f in files))

    def test_screen_verdicts(self):
        rows = screen.aggregate(common.load_ncu_csv(
            os.path.join(self.run_ncu, "derived", "ncu_metrics.csv")))
        by = {r["kernel"]: r for r in rows}
        self.assertEqual(screen.verdict(by["cast_image_kernel"]), "memory-bound")
        self.assertEqual(screen.verdict(by["sba::calc_jacobians_kernel"]), "compute-leaning")
        # time-weighting: single launch each, values pass through
        self.assertAlmostEqual(by["cast_image_kernel"]["mem_sol"], 85.0)
        screen.emit(rows, os.path.join(self.tmp, "out2"))
        self.assertTrue(os.path.isfile(os.path.join(self.tmp, "out2", "screen.csv")))

    def test_roofline_math(self):
        rows = roofline.aggregate(common.load_ncu_csv(
            os.path.join(self.run_ncu, "derived", "ncu_metrics.csv")))
        by = {r["kernel"]: r for r in rows}
        cast = by["cast_image_kernel"]
        # FLOPs = 1e6 + 5e5 + 2*2e6 = 5.5e6 ; DRAM = 2.5e6 B ; time = 100 us
        self.assertAlmostEqual(cast["flops"], 5.5e6)
        self.assertAlmostEqual(cast["ai_dram"], 5.5e6 / 2.5e6)
        self.assertAlmostEqual(cast["gflops"], 5.5e6 / 100e-6 / 1e9)
        files, warn = roofline.emit(rows, self.hw, os.path.join(self.tmp, "out3"))
        self.assertIsNone(warn)
        self.assertTrue(os.path.isfile(os.path.join(self.tmp, "out3", "fig_roofline.svg")))

    def test_bandwidth(self):
        rows = bandwidth.aggregate(common.load_ncu_csv(
            os.path.join(self.run_ncu, "derived", "ncu_metrics.csv")))
        by = {r["kernel"]: r for r in rows}
        cast = by["cast_image_kernel"]
        self.assertAlmostEqual(cast["bytes"], 2.5e6)
        self.assertAlmostEqual(cast["gbps"], 2.5e6 / 100e-6 / 1e9)
        files = bandwidth.emit(rows, self.hw, os.path.join(self.tmp, "out4"),
                               nsys_dir=self.run_nsys)
        self.assertTrue(os.path.isfile(os.path.join(self.tmp, "out4", "bandwidth.csv")))


if __name__ == "__main__":
    unittest.main(verbosity=2)
