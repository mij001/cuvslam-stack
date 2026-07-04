---
name: cuda-corruption-ld-library-path
description: System CUDA install is corrupt on this laptop; cuvslam needs LD_LIBRARY_PATH=~/.local/cuda-repair/lib until pacman reinstall
metadata: 
  node_type: memory
  type: project
  originSessionId: 83c42e24-7c04-403d-b5f5-5135ef6002f1
---

The CachyOS `cuda` pacman package on the dev laptop (host iNOMAL) has 118 corrupt files (zero-filled during the 2026-07-01 system update while /home was 99% full), including libnvJitLink.so.13 and libnvrtc.so.13. `import cuvslam` fails with "invalid ELF header" without the workaround.

**Why:** disk-full during pacman upgrade zeroed files under /opt/cuda.

**How to apply:** prefix workload commands with `LD_LIBRARY_PATH=$HOME/.local/cuda-repair/lib` (clean copies of the 13 corrupt .so files, extracted from /var/cache/pacman/pkg/cuda-13.3.1-1-x86_64.pkg.tar.zst with soname symlinks). Permanent fix (needs sudo, flagged to user as a task chip): `sudo pacman -S cuda`, then delete ~/.local/cuda-repair. Verify health with `LC_ALL=C pacman -Qkk cuda | grep -c "SHA256 checksum mismatch"` → 0. Related: [[cuvslam-stack-profiling-state]].
