#!/usr/bin/env bash
# fetch_datasets.sh — headless dataset fetcher for the profiling workloads.
# Downloads into ${CUVSLAM_DATASETS:-~/Projects/cuvslam_datasets}, resumable.
#
#   profiling/env/fetch_datasets.sh tum_office     # TUM fr3 long_office (1.5 GB) — loop-closure workload
#   profiling/env/fetch_datasets.sh euroc_v101     # EuRoC V1_01_easy   (1.6 GB) — steady-state workload
#   profiling/env/fetch_datasets.sh kitti          # prints manual instructions (registration required)
#   profiling/env/fetch_datasets.sh all
set -euo pipefail
ROOT="${CUVSLAM_DATASETS:-$HOME/Projects/cuvslam_datasets}"
mkdir -p "$ROOT"

fetch() { # url dest
    echo "[fetch] $1"
    curl -L --connect-timeout 20 --retry 3 -C - -o "$2" "$1"
}

tum_office() {
    local d="$ROOT/tum" f="rgbd_dataset_freiburg3_long_office_household"
    mkdir -p "$d"
    if [ -d "$d/$f" ]; then echo "[skip] $d/$f exists"; verify_tum_office "$d/$f"; return; fi
    fetch "https://cvg.cit.tum.de/rgbd/dataset/freiburg3/$f.tgz" "$d/$f.tgz"
    tar xzf "$d/$f.tgz" -C "$d" && rm "$d/$f.tgz"
    verify_tum_office "$d/$f"
    echo "[done] $d/$f"
}

# Integrity check (reproducibility): file count, total bytes, and sha256 of the
# index files, recorded from the 2026-07-02 download used in the committed report.
verify_tum_office() {
    local d="$1" ok=1
    [ "$(find "$d" -type f | wc -l)" = "5098" ] || { echo "[✗] file count mismatch"; ok=0; }
    [ "$(du -sb "$d" | cut -f1)" = "1574146525" ] || { echo "[✗] total size mismatch"; ok=0; }
    echo "9bc53b49d65090bf $(sha256sum "$d/rgb.txt" | cut -c1-16)" | awk '{exit $1!=$2}' || { echo "[✗] rgb.txt hash mismatch"; ok=0; }
    echo "7b45466238e892b8 $(sha256sum "$d/depth.txt" | cut -c1-16)" | awk '{exit $1!=$2}' || { echo "[✗] depth.txt hash mismatch"; ok=0; }
    echo "e255d5d9c5bab915 $(sha256sum "$d/groundtruth.txt" | cut -c1-16)" | awk '{exit $1!=$2}' || { echo "[✗] groundtruth.txt hash mismatch"; ok=0; }
    [ $ok -eq 1 ] && echo "[✓] dataset integrity verified (5098 files, sha256 index match)" \
                  || echo "[!] dataset differs from the version used in committed reports"
}

euroc_v101() {
    local d="$ROOT/euroc"
    mkdir -p "$d"
    if [ -d "$d/V1_01_easy/mav0" ]; then echo "[skip] $d/V1_01_easy exists"; return; fi
    # NOTE: robotics.ethz.ch is intermittently unreachable; retry later if this stalls.
    fetch "http://robotics.ethz.ch/~asl-datasets/ijrr_euroc_mav_dataset/vicon_room1/V1_01_easy/V1_01_easy.zip" \
          "$d/V1_01_easy.zip"
    unzip -q -o "$d/V1_01_easy.zip" -d "$d/V1_01_easy" && rm "$d/V1_01_easy.zip"
    echo "[done] $d/V1_01_easy"
}

kitti() {
    cat <<EOF
KITTI odometry requires registration — no scripted download.
  1. Register at https://www.cvlibs.net/datasets/kitti/eval_odometry.php
  2. Download 'odometry data set (grayscale, 22 GB)' and the calibration files
  3. Extract so that: \$CUVSLAM_DATASETS/dataset/sequences/06/image_0/*.png exists
     (ground-truth poses: \$CUVSLAM_DATASETS/dataset/poses/06.txt)
EOF
}

case "${1:-all}" in
    tum_office) tum_office ;;
    euroc_v101) euroc_v101 ;;
    kitti)      kitti ;;
    all)        tum_office; euroc_v101; kitti ;;
    *) echo "usage: $0 {tum_office|euroc_v101|kitti|all}"; exit 2 ;;
esac
