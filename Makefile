# cuVSLAM stack — umbrella project.
#
# Layout: the TOML runner lives at the repo root; NVIDIA cuVSLAM is a pinned git
# submodule in cuvslam/ (release 15.0 @ efdfbe56), overlaid at build time with
# the patches in patches/ (our Podman/CUDA-13 wheel tooling). This keeps the
# submodule pristine and the build reproducible anywhere:
#
#   make wheel   -> init submodule, apply patches, build the wheel (Podman)
#   make verify  -> install the wheel (setup_env) and run a TOML config
#
.PHONY: help init patch wheel verify check clean unpatch all
.DEFAULT_GOAL := help

TOML := configs/kitti_eval.toml
DIST := $(CURDIR)/cuvslam/dist

help:
	@echo "make wheel   - init the cuvslam submodule, apply patches/, build the wheel (Podman, CUDA 13, py3.10)"
	@echo "make verify  - install the built wheel via setup_env and run $(TOML)"
	@echo "make check   - install the built wheel and validate $(TOML)"
	@echo "make clean   - remove the runner venv"
	@echo "make unpatch - revert the submodule overlay back to pristine efdfbe56"
	@echo "make all     - wheel + verify"

# Fetch the pinned cuVSLAM source.
init:
	git submodule update --init cuvslam

# Idempotently overlay the build tooling onto the pinned submodule.
patch: init
	@if [ ! -f cuvslam/build_wheel.sh ]; then \
	  ( cd cuvslam && git apply ../patches/*.patch ) && chmod +x cuvslam/build_wheel.sh && echo "patches applied"; \
	else echo "patches already applied"; fi

# Build libcuvslam (cmake) + the scikit-build-core wheel -> cuvslam/dist/.
wheel: patch
	cd cuvslam && ./build_wheel.sh

# Install the freshly built wheel into the runner venv and run a config.
verify:
	WHEEL="$$(ls -t $(DIST)/cuvslam-*.whl | head -1)" ./setup_env.sh
	./cuvslam_venv/bin/python run.py $(TOML)

check:
	WHEEL="$$(ls -t $(DIST)/cuvslam-*.whl | head -1)" ./setup_env.sh
	./cuvslam_venv/bin/python run.py $(TOML) --check

clean:
	./cleanup_env.sh

# Revert cuvslam/ to the pristine pinned commit (drops the overlay + build output).
unpatch:
	-cd cuvslam && git checkout -- . && git clean -fdq

all: wheel verify
