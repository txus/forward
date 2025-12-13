.SHELLFLAGS := -eu -o pipefail -c

.PHONY: all
all:
	@cmake -S . -B build -GNinja -DCMAKE_BUILD_TYPE=Debug && cmake --build build --parallel --

.PHONY: release
release:
	@cmake -S . -B build -GNinja -DCMAKE_BUILD_TYPE=Release && cmake --build build --parallel --

.PHONY: app
app:
	@cmake --build build --target forward
	@./build/apps/forward

.PHONY: tensor
tensor:
	@cmake --build build --target test_tensor
	@ctest --test-dir build -R "^Tensor" --output-on-failure

.PHONY: nn
nn:
	@cmake --build build --target test_nn
	@ctest --test-dir build -R "^NN" --output-on-failure

.PHONY: llama
llama:
	@cmake --build build --target test_llama
	@ctest --test-dir build -R "^Llama" --output-on-failure

.PHONY: forward
forward:
	@cmake --build build --target test_forward
	@ctest --test-dir build -R "^Forward" --output-on-failure

.PHONY: test
test:
	@cmake --build build
	@ctest --test-dir build --output-on-failure


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

