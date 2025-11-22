SHELL = /bin/bash
.SHELLFLAGS := -eu -o pipefail -c

.PHONY: all
all:
	@cmake -S . -B build && cmake --build build --parallel --

.PHONY: release
release:
	@cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build --parallel --

.PHONY: app
app:
	@cd build && make forward && cd .. && ./build/apps/forward

.PHONY: test
tensor:
	@cd build && make test_tensor && ./tests/tensor/test_tensor

.PHONY: llama
llama:
	@cd build && make test_llama && ./tests/llama/test_llama

.PHONY: forward
forward:
	@cd build && make test_forward && ./tests/forward/test_forward

.PHONY: test
test: tensor llama forward
	@cd build && ./tests/tensor/test_tensor && ./tests/llama/test_llama && ./tests/forward/test_forward


.PHONY: lint
lint:
	@./scripts/lint.sh

.PHONY: lint-fix
lint-fix:
	@./scripts/lint.sh --fix

.PHONY: format
format:
	@echo "Formatting C++ files..."
	@find src include tests apps -type f \( -name "*.cpp" -o -name "*.hpp" \) -exec clang-format -i {} +
	@echo "✓ Formatting complete!"

.PHONY: clean
clean:
	@rm -rf build*/ || true

