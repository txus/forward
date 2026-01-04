.SHELLFLAGS := -eu -o pipefail -c

.PHONY: all
all:
	@cmake --preset ninja-nvcc -DCMAKE_BUILD_TYPE=Debug && cmake --build build --parallel --

.PHONY: rebuild
rebuild:
	@cmake --preset ninja-nvcc -DCMAKE_BUILD_TYPE=Debug

.PHONY: dx
dx:
	@cmake --preset ninja-clangd && cmake --build --preset ninja-clangd -t tensor_cpu

.PHONY: release
release:
	@cmake --preset ninja-nvcc -DCMAKE_BUILD_TYPE=Release && cmake --build build --parallel --

.PHONY: prepare_profile
prepare_profile:
	@cmake --preset ninja-nvcc -DCMAKE_BUILD_TYPE=Release
	@cmake --build build --parallel --target forward

.PHONY: profile
profile:
	@cmake --build build --parallel --target test_tensor_cuda
	@echo 'sudo ncu --kernel-name "add_kernel" ctest --test-dir build -R "^TensorCUDATest.AddBF16"'

.PHONY: app
app:
	@cmake --build build --target forward
	@./build/apps/forward

.PHONY: inspect
inspect:
	@cmake --build build --target inspect
	@./build/apps/inspect

.PHONY: tensor
tensor:
	@cmake --build build --target tensor
	@ctest --test-dir build -R "^TensorCPU" --output-on-failure
	@ctest --test-dir build -R "^TensorCUDA" --output-on-failure

.PHONY: tensor_cpu
tensor_cpu:
	@cmake --build build --target test_tensor_cpu
	@ctest --test-dir build -R "^TensorCPU" --output-on-failure

.PHONY: tensor_cuda
tensor_cuda:
	@cmake --build build --target test_tensor_cuda
	@ctest --test-dir build -R "^TensorCUDA" --output-on-failure

.PHONY: nn
nn:
	@cmake --build build --target nn
	@ctest --test-dir build -R "^NNCPU" --output-on-failure
	@ctest --test-dir build -R "^NNCUDA" --output-on-failure

.PHONY: nn_cpu
nn_cpu:
	@cmake --build build --target test_nn_cpu
	@ctest --test-dir build -R "^NNCPU" --output-on-failure

.PHONY: nn_cuda
nn_cuda:
	@cmake --build build --target test_nn_cuda
	@ctest --test-dir build -R "^NNCUDA" --output-on-failure

.PHONY: llama
llama:
	@cmake --build build --target test_llama
	@ctest --test-dir build -R "^Llama" --output-on-failure

.PHONY: llama_cuda
llama_cuda:
	@cmake --build build --target test_llama
	@ctest --test-dir build -R "^LlamaCUDA" --output-on-failure

.PHONY: forward
forward:
	@cmake --build build --target test_forward
	@ctest --test-dir build -R "^Forward" --output-on-failure

.PHONY: test
test:
	@cmake --build build
	@ctest --test-dir build --output-on-failure


.PHONY: prepare_benchmark
prepare_benchmark:
	@cmake --preset ninja-nvcc -DCMAKE_BUILD_TYPE=Release -DFUSED_ROPE=ON

.PHONY: benchmark_tensor
benchmark: prepare_benchmark
	@cmake --build build --target bm_tensor
	./build/benchmarks/tensor/bm_tensor

.PHONY: benchmark_llama
benchmark_llama: prepare_benchmark
	@cmake --build build --target bm_llama
	./build/benchmarks/llama/bm_llama

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

