.SHELLFLAGS := -eu -o pipefail -c

# Pick the preset by host OS. Override with `make ... PRESET=linux` if needed.
PRESET ?= $(shell uname -s | grep -q Darwin && echo macos || echo linux)
# Space-separated labels -> repeated -L flags (ctest ANDs them).
LBL    := $(foreach l,$(LABELS),-L $(l))

.PHONY: configure
configure:
	@cmake --preset $(PRESET) $(ARGS)

.PHONY: build
build: configure
	@cmake --build build -j

.PHONY: release
release:
	@cmake --preset $(PRESET) -DCMAKE_BUILD_TYPE=Release
	@cmake --build build -j

# Run a subset of tests by label: make test LABELS=tensor
#                                 make test LABELS="tensor metal"
#                                 make test            (everything buildable here)
.PHONY: test
test: configure
	@cmake --build build -j
	@ctest --test-dir build $(LBL) --output-on-failure

# Pattern rules can't be listed in .PHONY (make ignores it), but they're
# effectively phony here: no file named run-* or bench-* ever exists in the
# repo root, so the recipe always runs.

# Build + run an app target: make run-forward / run-test_metal / run-inspect
run-%: configure
	@cmake --build build --target $*
	@./build/apps/$*

# Build + run a benchmark in RelWithDebInfo: make bench-tensor / bench-llama
# (bench-llama needs the CUDA backend; bm_llama isn't built on macOS.)
bench-%:
	@cmake --preset $(PRESET)-bench
	@cmake --build build-bench --target bm_$*
	@./build-bench/bench/bm_$*

# Build a profiling binary (RelWithDebInfo) and print the ncu command to run it.
# CUDA-only: targets a CUDA test and Nsight Compute (ncu).
.PHONY: profile
profile:
	@cmake --preset $(PRESET)-bench
	@cmake --build build-bench --target test_tensor_cuda
	@echo 'sudo ncu --kernel-name "add_kernel" ctest --test-dir build-bench -R "^TensorCUDA"'

.PHONY: lint
lint:
	@./scripts/lint.sh

.PHONY: format
format:
	@echo "Formatting C++ files..."
	@find src include tests apps -type f \( -name "*.cpp" -o -name "*.hpp" \) -exec clang-format -i {} +
	@echo "✓ Formatting complete!"

.PHONY: clean
clean:
	@rm -rf build*/ || true
