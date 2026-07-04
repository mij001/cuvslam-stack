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
from analysis import bandwidth, build_dag, classify, common, roofline, screen, stages  # noqa: E402


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


class TestClassify(unittest.TestCase):
    HW = {"memory": {"l2_bytes": 524288}}

    def _row(self, **kw):
        base = {"kernel": "k", "stage": "other", "time_s": 1e-3, "launches": 1,
                "mem_sol": float("nan"), "comp_sol": float("nan"),
                "dram_sol": float("nan"), "l1_hit": float("nan"),
                "l2_hit": float("nan"), "occ": float("nan"),
                "sect_ld": float("nan"), "sect_st": float("nan"),
                "lfmr": float("nan"), "mpki": float("nan"),
                "dram_bytes_per_launch": float("nan"), "ai_dram": float("nan")}
        for n, _, _ in screen.STALLS:
            base[f"stall_{n}"] = float("nan")
        base.update(kw)
        return base

    def test_decision_tree(self):
        cases = [
            # compute-bound: high comp SoL
            (dict(comp_sol=80.0, mem_sol=20.0), "G5-compute"),
            # bandwidth-bound: DRAM saturated, caches useless
            (dict(mem_sol=85.0, comp_sol=10.0, dram_sol=75.0, lfmr=0.6), "G1-bandwidth"),
            # coalescing: memory-limited + scattered
            (dict(mem_sol=50.0, comp_sol=5.0, dram_sol=20.0, sect_ld=20.0, lfmr=0.5),
             "G2-coalescing"),
            # L2-reuse: memory-limited but L2 absorbs (low LFMR)
            (dict(mem_sol=55.0, comp_sol=10.0, dram_sol=15.0, lfmr=0.05, sect_ld=4.0),
             "G3-l2-reuse"),
            # latency at low occupancy, long-scoreboard dominant
            (dict(mem_sol=10.0, comp_sol=2.0, dram_sol=8.0, occ=5.0, sect_ld=4.0,
                  lfmr=0.5, stall_long_scoreboard=8.0, stall_wait=0.5,
                  dram_bytes_per_launch=2e7), "G4-latency"),
            # dependency-bound: 'wait' dominant, low occupancy, memory NOT the wall
            (dict(mem_sol=5.0, comp_sol=1.0, dram_sol=4.0, occ=16.0,
                  stall_wait=2.4, stall_short_scoreboard=1.3,
                  stall_long_scoreboard=0.6), "G7-dependency"),
            # on-chip: MIO dominant, DRAM unsaturated
            (dict(mem_sol=30.0, comp_sol=20.0, dram_sol=10.0,
                  stall_mio_throttle=5.0, stall_long_scoreboard=1.0), "G6-onchip"),
            # nothing dominant
            (dict(mem_sol=5.0, comp_sol=3.0, dram_sol=2.0, occ=60.0), "G0-nosignal"),
        ]
        for kw, want in cases:
            got = classify.classify_kernel(self._row(**kw), self.HW)["class"]
            self.assertEqual(got, want, f"{kw} -> {got}, want {want}")

    def test_pim_affinity_cold_scan(self):
        r = self._row(stage="slam_loop", sect_ld=18.0, lfmr=0.14, occ=3.0,
                      mem_sol=9.0, dram_sol=7.0, stall_long_scoreboard=6.0,
                      dram_bytes_per_launch=2.35e7)
        c = classify.classify_kernel(r, self.HW)
        aff, sub = classify.pim_affinity(c["class"], "cold-persistent", c)
        self.assertEqual(aff, "strong")
        self.assertIn("ISP", sub)

    def test_load_features_from_data_dir(self):
        tmp = tempfile.mkdtemp()
        try:
            with open(os.path.join(tmp, "screen.csv"), "w") as fh:
                fh.write("kernel,stage,verdict,launches,time_ms,mem_sol_pct,comp_sol_pct,"
                         "dram_sol_pct,l1_hit_pct,l2_hit_pct,occupancy_pct,"
                         "sectors_per_req_ld,sectors_per_req_st,lfmr_gpu,mpki_gpu,"
                         "dram_bytes_per_launch,stall_long_scoreboard\n")
                fh.write("k1,preprocess,memory-bound,4,1.5,85,10,75,50,40,80,4,4,0.6,12,2e6,20\n")
            rows = classify.load_features(tmp)
            self.assertEqual(rows[0]["kernel"], "k1")
            self.assertAlmostEqual(rows[0]["lfmr"], 0.6)
            got = classify.classify_kernel(rows[0], self.HW)["class"]
            self.assertEqual(got, "G1-bandwidth")
        finally:
            shutil.rmtree(tmp)


class TestLocality(unittest.TestCase):
    def _trace(self):
        # kernel A, two launches over the SAME 8 sectors (64B apart => 2
        # sectors per 2-address access below), then one scattered access.
        # kernel name includes a full demangled signature WITH SPACES (as the
        # real tool prints) — the LAUNCH regex must span it
        lines = ["MEMTRACE: CTX 0x1 - LAUNCH - Kernel pc 0x1 - Kernel name "
                 "cuvslam::cuda::kernA(unsigned char const*, unsigned long) "
                 "- grid launch id 0 - grid size 1,1,1 - block size 32,1,1 "
                 "- nregs 1 - shmem 0 - cuda stream id 1"]
        base = 0x10000
        for rep in range(2):           # touch sectors twice -> reuse
            for i in range(4):
                a0, a1 = base + i * 64, base + i * 64 + 32
                lines.append(f"MEMTRACE: CTX 0x1 - grid_launch_id 0 - CTA 0,0,0 "
                             f"- warp 0 - LDG.E - {hex(a0)} {hex(a1)} " + "0x0 " * 30)
        lines.append("MEMTRACE: CTX 0x1 - LAUNCH - Kernel pc 0x1 - Kernel name "
                     "cuvslam::cuda::kernA(unsigned char const*, unsigned long) "
                     "- grid launch id 1 - grid size 1,1,1 - block size 32,1,1 "
                     "- nregs 1 - shmem 0 - cuda stream id 1")
        for i in range(4):
            a0 = base + i * 64
            lines.append(f"MEMTRACE: CTX 0x1 - grid_launch_id 1 - CTA 0,0,0 "
                         f"- warp 0 - LDG.E - {hex(a0)} " + "0x0 " * 31)
        return "\n".join(lines) + "\n"

    def test_locality_pipeline(self):
        from analysis import locality
        tmp = tempfile.mkdtemp()
        try:
            p = os.path.join(tmp, "trace.txt")
            open(p, "w").write(self._trace())
            kernels = locality.analyze(p)
            self.assertIn("kernA", kernels)
            ks = kernels["kernA"]
            self.assertEqual(ks.launches, 2)
            # launch 0 footprint: 4 addr-pairs at 64B stride = 8 sectors
            self.assertEqual(ks.footprints[0], 8)
            # second pass over same sectors -> reuse distances recorded
            self.assertGreater(sum(v for k, v in ks.reuse_hist.items()
                                   if isinstance(k, int) and k >= 0), 0)
            # launch 1 touches 4 of the 8 sectors -> Jaccard 0.5
            self.assertAlmostEqual(ks.overlaps[0], 0.5)
            # every warp access touched 1-2 unique sectors -> coalesced
            cdf = locality.hit_cdf(ks)
            self.assertGreater(cdf[64 * 1024], 0.3)   # reuse fits any cache
            files = locality.emit(kernels, tmp)
            self.assertTrue(all(os.path.isfile(f) for f in files))
        finally:
            shutil.rmtree(tmp)


class TestAttribution(unittest.TestCase):
    def test_journal_parse_and_tagging(self):
        from analysis import attribution
        tmp = tempfile.mkdtemp()
        try:
            j = os.path.join(tmp, "journal.csv")
            with open(j, "w") as fh:
                fh.write("#cuvslam-alloc-log,v1\n"
                         "M,7f0000000000-7f0000100000 r-xp 00020000 08:01 1 /lib/libcuvslam.so\n"
                         "A,100,0x10000,4096,GPUOnlyArray,0x7f0000030000\n"
                         "F,200,0x10000\n")
            maps, allocs, frees = attribution.parse_journal(j)
            self.assertEqual(len(maps), 1)
            self.assertEqual(allocs[0]["bytes"], 4096)
            self.assertEqual(frees[0]["ptr"], 0x10000)
            mod, off = attribution._rebase(0x7f0000030000, maps)
            self.assertEqual((mod, off), ("/lib/libcuvslam.so", 0x50000))
        finally:
            shutil.rmtree(tmp)

    def test_owner_skips_plumbing(self):
        from analysis import attribution
        resolved = {
            0x1: [("cuvslam::cuda::GPUOnlyArray<float>::GPUOnlyArray(unsigned long)",
                   "/cuvslam/libs/cuda_modules/cuda_helper.h:196")],
            0x2: [("std::make_unique<Foo>()", "/usr/include/c++/11/memory:1")],
            0x3: [("cuvslam::sba::SchurComplementBundlerGpu::Impl::Impl()",
                   "/cuvslam/libs/sba/schur_complement_bundler_gpu.cpp:59")],
        }
        func, site = attribution.owner_of([0x1, 0x2, 0x3], resolved)
        self.assertIn("SchurComplement", func)
        self.assertEqual(attribution.tag_of(func, site), "ba_linear_system")

    def test_join_lifetime_and_unmapped(self):
        from analysis import attribution
        tmp = tempfile.mkdtemp()
        try:
            # buffer at 0x10000 is keyframe_descriptors during launch 0,
            # freed and re-allocated as ba_linear_system before launch 1;
            # 0x90000 is never allocated -> unmapped
            table = os.path.join(tmp, "alloc_table.csv")
            with open(table, "w") as fh:
                fh.write("t_us,ptr,bytes,kind,tag,owner_func,owner_site\n"
                         "1,0x10000,4096,GPUArray,keyframe_descriptors,f,s\n"
                         "2,0x10000,,FREE,,,\n"
                         "3,0x10000,4096,GPUImage,ba_linear_system,f,s\n")
            sidecar = os.path.join(tmp, "sidecar.csv")
            with open(sidecar, "w") as fh:
                fh.write("#mem-trace-alloc-events,v1\n"
                         "ALLOC,0,0x10000,4096\n"
                         "FREE,1,0x10000\n"
                         "ALLOC,1,0x10000,4096\n")
            trace = os.path.join(tmp, "trace.txt")
            launch = ("MEMTRACE: CTX 0x1 - LAUNCH - Kernel pc 0x1 - Kernel name "
                      "cuvslam::cuda::kern{n}(float*) - grid launch id {g} - grid size "
                      "1,1,1 - block size 32,1,1 - nregs 1 - shmem 0 - cuda stream id 1")
            acc = ("MEMTRACE: CTX 0x1 - grid_launch_id {g} - CTA 0,0,0 - warp 0 "
                   "- LDG.E - {a} " + "0x0 " * 31)
            with open(trace, "w") as fh:
                fh.write("\n".join([
                    launch.format(n="A", g=0), acc.format(g=0, a=hex(0x10040)),
                    acc.format(g=0, a=hex(0x90000)),
                    launch.format(n="B", g=1), acc.format(g=1, a=hex(0x10040)),
                ]) + "\n")
            out = os.path.join(tmp, "out")
            attribution.main(["join", trace, table, sidecar, "--out", out])
            rows = {}
            with open(os.path.join(out, "attribution.csv")) as fh:
                next(fh)
                for line in fh:
                    k, tag, *_rest = line.strip().split(",")
                    rows[(k, tag)] = _rest
            self.assertIn(("kernA", "keyframe_descriptors"), rows)
            self.assertIn(("kernA", "unmapped"), rows)
            self.assertIn(("kernB", "ba_linear_system"), rows)
            self.assertNotIn(("kernB", "keyframe_descriptors"), rows)
        finally:
            shutil.rmtree(tmp)

    def test_join_access_cap_early_stop(self):
        # a huge single-kernel trace must be bounded by the per-kernel cap and
        # early-stop long before EOF; trailing records go uncounted
        from analysis import attribution
        tmp = tempfile.mkdtemp()
        try:
            table = os.path.join(tmp, "alloc_table.csv")
            with open(table, "w") as fh:
                fh.write("t_us,ptr,bytes,kind,tag,owner_func,owner_site\n"
                         "1,0x10000,65536,GPUArray,keyframe_descriptors,f,s\n")
            sidecar = os.path.join(tmp, "sidecar.csv")
            with open(sidecar, "w") as fh:
                fh.write("ALLOC,0,0x10000,65536\n")
            trace = os.path.join(tmp, "trace.txt")
            with open(trace, "w") as fh:
                fh.write("MEMTRACE: CTX 0x1 - LAUNCH - Kernel pc 0x1 - Kernel name "
                         "cuvslam::cuda::st_kernel(float*) - grid launch id 0 - grid "
                         "size 1,1,1 - block size 32,1,1 - nregs 1 - shmem 0 - cuda "
                         "stream id 1\n")
                acc = ("MEMTRACE: CTX 0x1 - grid_launch_id 0 - CTA 0,0,0 - warp 0 "
                       "- LDG.E - {a} " + "0x0 " * 31 + "\n")
                for i in range(100000):
                    fh.write(acc.format(a=hex(0x10000 + (i % 512) * 32)))
            out = os.path.join(tmp, "out")
            attribution.main(["join", trace, table, sidecar, "--out", out,
                              "--max-accesses-per-kernel", "1000"])
            with open(os.path.join(out, "attribution.csv")) as fh:
                next(fh)
                row = next(fh).strip().split(",")
            # kernel, tag, warp_accesses, ... — capped at 1000, not 100000
            self.assertEqual(row[0], "st_kernel")
            self.assertEqual(row[1], "keyframe_descriptors")
            self.assertEqual(int(row[2]), 1000)
        finally:
            shutil.rmtree(tmp)

    def test_join_memory_space_buckets(self):
        # LDS/STS -> shared_onchip, LDL/STL -> local_spill, never the live set
        from analysis import attribution
        tmp = tempfile.mkdtemp()
        try:
            table = os.path.join(tmp, "alloc_table.csv")
            with open(table, "w") as fh:
                fh.write("t_us,ptr,bytes,kind,tag,owner_func,owner_site\n"
                         "1,0x0,4096,GPUArray,icp_state,f,s\n")
            sidecar = os.path.join(tmp, "sidecar.csv")
            with open(sidecar, "w") as fh:
                fh.write("ALLOC,0,0x0,4096\n")
            trace = os.path.join(tmp, "trace.txt")
            launch = ("MEMTRACE: CTX 0x1 - LAUNCH - Kernel pc 0x1 - Kernel name "
                      "cuvslam::cuda::kernS(float*) - grid launch id 0 - grid size "
                      "1,1,1 - block size 32,1,1 - nregs 1 - shmem 0 - cuda stream id 1")
            acc = ("MEMTRACE: CTX 0x1 - grid_launch_id 0 - CTA 0,0,0 - warp 0 "
                   "- {op} - {a} " + "0x0 " * 31)
            with open(trace, "w") as fh:
                fh.write("\n".join([
                    launch,
                    acc.format(op="LDS", a="0x100"),      # in-range of the alloc,
                    acc.format(op="STL.128", a="0x200"),  # but space wins
                    acc.format(op="LDG.E", a="0x300"),
                ]) + "\n")
            out = os.path.join(tmp, "out")
            attribution.main(["join", trace, table, sidecar, "--out", out])
            tags = set()
            with open(os.path.join(out, "attribution.csv")) as fh:
                next(fh)
                for line in fh:
                    tags.add(line.split(",")[1])
            self.assertEqual(tags, {"shared_onchip", "local_spill", "icp_state"})
        finally:
            shutil.rmtree(tmp)


if __name__ == "__main__":
    unittest.main(verbosity=2)
