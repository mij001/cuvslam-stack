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
.PHONY: help init patch wheel verify check clean unpatch all
.DEFAULT_GOAL := help

TOML := configs/kitti_eval.toml
DIST := $(CURDIR)/cuvslam_src/dist

help:
	@echo "make wheel   - init the cuvslam_src submodule, apply patches/, build the wheel (Podman, CUDA 13, py3.10)"
	@echo "make verify  - install the built wheel via setup_env and run $(TOML)"
	@echo "make check   - install the built wheel and validate $(TOML)"
	@echo "make clean   - remove the runner venv"
	@echo "make unpatch - revert the submodule overlay back to pristine efdfbe56"
	@echo "make all     - wheel + verify"

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
