#!/usr/bin/env bash
# fetch_paper_datasets.sh — download the cuVSLAM-paper datasets missing from
# sda2, with aria2c. Idempotent (skips what's present). Run on the workstation
# after making /mnt/data writable:
#   sudo -n umount /mnt/data; sudo -n mount -t ntfs3 -o rw,force /dev/sda2 /mnt/data
#
#   ./fetch_paper_datasets.sh tum        # 9 TUM fr3 (paper Table 4), skip-if-present
#   ./fetch_paper_datasets.sh icl        # ICL-NUIM 8 trajectories (paper Mono-Depth)
#   ./fetch_paper_datasets.sh all        # tum + icl  (TartanAir is separate — see prep_tartanair.py)
set -uo pipefail
ROOT="${CUVSLAM_DATA2:-/mnt/data}"

# sda2 is fstab-ro and pre-existing dirs are root-owned; re-assert a uid-mapped
# rw mount so downloads into existing dirs (e.g. TUM_RGBD/extracted) succeed.
if ! touch "$ROOT/.wtest" 2>/dev/null; then
    sudo -n umount "$ROOT" 2>/dev/null || true
    sudo -n mount -t ntfs3 -o rw,force,uid=1000,gid=1000,umask=022 /dev/sda2 "$ROOT" 2>/dev/null || true
fi
rm -f "$ROOT/.wtest" 2>/dev/null || true
DL() { aria2c -x8 -s8 -c --auto-file-renaming=false -d "$1" "$2" || echo "[!] failed: $2"; }

tum() {
    local base="https://cvg.cit.tum.de/rgbd/dataset/freiburg3"
    local d="$ROOT/TUM_RGBD/extracted"; mkdir -p "$d"
    for s in cabinet long_office_household nostructure_texture_far \
             nostructure_texture_near_withloop sitting_halfsphere sitting_xyz \
             structure_texture_far structure_texture_near teddy; do
        local name="rgbd_dataset_freiburg3_$s"
        [ -d "$d/$name" ] && { echo "[skip] $name"; continue; }
        echo "=== TUM $name"
        DL "$d" "$base/$name.tgz"
        tar xzf "$d/$name.tgz" -C "$d" && rm -f "$d/$name.tgz"
    done
}

icl() {
    # ICL-NUIM TUM-format (Handa). RGB-D tarball + GT (.gt.freiburg → groundtruth.txt).
    local host="https://www.doc.ic.ac.uk/~ahanda"
    local d="$ROOT/ICL-NUIM"; mkdir -p "$d"
    # living-room traj0-3  and  office traj0-3
    declare -A GT=( [living_room_traj0_frei_png]=livingRoom0.gt.freiburg
                    [living_room_traj1_frei_png]=livingRoom1.gt.freiburg
                    [living_room_traj2_frei_png]=livingRoom2.gt.freiburg
                    [living_room_traj3_frei_png]=livingRoom3.gt.freiburg
                    [traj0_frei_png]=traj0.gt.freiburg
                    [traj1_frei_png]=traj1.gt.freiburg
                    [traj2_frei_png]=traj2.gt.freiburg
                    [traj3_frei_png]=traj3.gt.freiburg )
    for seq in "${!GT[@]}"; do
        [ -d "$d/$seq" ] && { echo "[skip] $seq"; continue; }
        echo "=== ICL-NUIM $seq"
        DL "$d" "$host/$seq.tar.gz"
        mkdir -p "$d/$seq"
        tar xzf "$d/$seq.tar.gz" -C "$d/$seq" && rm -f "$d/$seq.tar.gz"
        # GT (VaFRIC path); TUM format already (timestamp tx ty tz qx qy qz qw)
        DL "$d/$seq" "$host/VaFRIC/${GT[$seq]}"
        [ -f "$d/$seq/${GT[$seq]}" ] && cp "$d/$seq/${GT[$seq]}" "$d/$seq/groundtruth.txt"
    done
    # The _frei_png release ships associations.txt (ts_d depth/N.png ts_rgb
    # rgb/N.png) + groundtruth.txt, but not the rgb.txt/depth.txt the tum source
    # needs. Generate them from associations.txt for every ICL sequence.
    icl_index
}

icl_index() {
    local d="$ROOT/ICL-NUIM"
    for a in "$d"/*/associations.txt; do
        [ -f "$a" ] || continue
        local sd; sd="$(dirname "$a")"
        awk '{print $3, $4}' "$a" > "$sd/rgb.txt"
        awk '{print $1, $2}' "$a" > "$sd/depth.txt"
        echo "[✓] ICL index: $(basename "$sd") ($(wc -l < "$sd/rgb.txt") frames)"
    done
}

case "${1:-all}" in
    tum) tum ;;
    icl) icl ;;
    all) tum; icl ;;
    *) echo "usage: $0 {tum|icl|all}"; exit 2 ;;
esac
echo "[done] $ROOT"; df -h "$ROOT" | tail -1
