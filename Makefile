SHELL = /bin/bash
.SHELLFLAGS := -eu -o pipefail -c

.PHONY: all
all:
	@cmake -S . -B build && cmake --build build --parallel --

.PHONY: release
release:
	@cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build --parallel --

.PHONY: test
test:
	@cd build && make test

.PHONY: clean
clean:
	@rm -rf build*/ || true

