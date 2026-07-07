# cuVSLAM stack — umbrella project.
#
# Layout: the TOML runner lives at the repo root; NVIDIA cuVSLAM is a pinned git
# submodule in cuvslam_src/ (release 15.0 @ efdfbe56), overlaid at build time
# with the patches in patches/ (our Podman/CUDA-13 wheel tooling). This keeps the
# submodule pristine and the build reproducible anywhere. The submodule is named
# cuvslam_src (not cuvslam) so it never shadows the installed `cuvslam` package.
#
#   make wheel   -> init submodule, apply patches, build the wheel (Podman)
#   make verify  -> install the wheel (setup_env) and run a TOML config
#
.PHONY: help init patch wheel verify check clean unpatch all \
        build configs validate profile analyze figures site
.DEFAULT_GOAL := help

TOML := configs/kitti_eval.toml
DIST := $(CURDIR)/cuvslam_src/dist

# ── phase parameters (override on the command line) ──────────────────────────
PY    ?= ./cuvslam_venv/bin/python
DATA  ?= /mnt/data
HW    ?= profiling/hw/dellworkstation_sm89.toml
CFG   ?= configs/base/kitti06_stereo_slam.toml
SCOPE ?= accuracy

help:
	@echo "BUILD phase (wheel + venv — no GPU profiling here):"
	@echo "  make build    - wheel (submodule+patches, Podman) + setup_env venv"
	@echo "  make wheel    - just the wheel"
	@echo "  make verify   - install the built wheel and run $(TOML)"
	@echo "  make check    - install the built wheel and validate $(TOML)"
	@echo "  make clean    - remove the runner venv;  make unpatch - pristine submodule"
	@echo ""
	@echo "CONFIG phase:"
	@echo "  make configs  - bases from DATA=$(DATA) + all mutations -> configs/generated/"
	@echo ""
	@echo "PROFILING phase (workstation, GPU):"
	@echo "  make validate - validation regime: configs x {plain,nsys,ncu,nvbit} (SCOPE=$(SCOPE))"
	@echo "  make profile  - cohesive pipeline on CFG=$(CFG): nsys->window->ncu->nvbit->analyses"
	@echo ""
	@echo "ANALYSIS phase (anywhere):"
	@echo "  make analyze  - substrate candidacy + dynamics from the classification tables"
	@echo "  make figures  - PNG counterparts for every committed artifact"
	@echo "  make site     - figures + the browsable results site (site/index.html)"

# ── phases ────────────────────────────────────────────────────────────────────
build: wheel
	./setup_env.sh

configs:
	-python3 scripts/gen_base_configs.py --root $(DATA)
	python3 scripts/mutate_configs.py --select all

validate:
	scripts/validation_regime.sh $(SCOPE)

profile:
	$(PY) profiling/regime.py --config $(CFG) --hw $(HW)

analyze:
	$(PY) profiling/analysis/substrate.py profiling/reports/*/data/classification.csv \
	    --agreement profiling/reports/2026-07-04_campaign/class_agreement.csv \
	    --out reports/2026-07-07_substrate

figures:
	$(PY) viz/make_figures.py

site: figures
	$(PY) viz/build_site.py

# Fetch the pinned cuVSLAM source. Skip LFS smudge -- the build needs only the
# source (libs/, python/, CMake), not the example media stored in Git LFS, and
# pulling LFS blobs is slow and can stall on a fresh clone.
init:
	GIT_LFS_SKIP_SMUDGE=1 git submodule update --init cuvslam_src

# Idempotently overlay the build tooling onto the pinned submodule.
patch: init
	@if [ ! -f cuvslam_src/build_wheel.sh ]; then \
	  ( cd cuvslam_src && git apply ../patches/*.patch ) && chmod +x cuvslam_src/build_wheel.sh && echo "patches applied"; \
	else echo "patches already applied"; fi

# Build libcuvslam (cmake) + the scikit-build-core wheel -> cuvslam_src/dist/.
wheel: patch
	cd cuvslam_src && ./build_wheel.sh

# Install the freshly built wheel into the runner venv and run a config.
verify:
	WHEEL="$$(ls -t $(DIST)/cuvslam-*.whl | head -1)" ./setup_env.sh
	./cuvslam_venv/bin/python run.py $(TOML)

check:
	WHEEL="$$(ls -t $(DIST)/cuvslam-*.whl | head -1)" ./setup_env.sh
	./cuvslam_venv/bin/python run.py $(TOML) --check

clean:
	./cleanup_env.sh

# Revert cuvslam_src/ to the pristine pinned commit (drops the overlay + build output).
unpatch:
	-cd cuvslam_src && git checkout -- . && git clean -fdq

all: wheel verify
