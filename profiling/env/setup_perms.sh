#!/usr/bin/env bash
# Grant non-root access to GPU performance counters so Nsight Compute can read them
# (otherwise ncu fails with ERR_NVGPUCTRPERM). One-time; needs sudo + a reboot.
#
#   sudo profiling/env/setup_perms.sh
#
# Check whether it's already in effect (no reboot needed if this prints 0):
#   grep RmProfilingAdminOnly /proc/driver/nvidia/params
set -euo pipefail
[[ $EUID -eq 0 ]] || exec sudo bash "$0" "$@"

CONF=/etc/modprobe.d/nvidia-profiler.conf
LINE="options nvidia NVreg_RestrictProfilingToAdminUsers=0"
if grep -qxF "$LINE" "$CONF" 2>/dev/null; then
  echo "[i] $CONF already grants profiling permissions."
else
  echo "$LINE" > "$CONF"
  echo "[✓] wrote $CONF"
fi
update-initramfs -u 2>&1 | tail -2 || echo "[!] update-initramfs failed (check distro)"
echo "[i] Reboot for it to take effect. Verify after reboot with:"
echo "    grep RmProfilingAdminOnly /proc/driver/nvidia/params   # want: 0"
