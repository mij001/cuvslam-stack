# Slice-3 locality data (committed derived tables)

Per-kernel locality from NVBit mem_trace, produced by `analysis/locality.py`
on the RTX 2000 Ada (driver 575.64.05 / CUDA 12.9, clocks locked). See
../FINDINGS.md. Raw traces (GB-scale zstd) stay on the workstation at
~/slice3/; these CSVs are the portable result.

- tum_odom/    front-end steady-state (streaming kernels): flat reuse CDF
- tum_sttrack/, kitti00_sttrack/  loop-closure scan at room / street scale
