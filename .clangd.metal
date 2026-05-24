# Metal / macOS clangd config.
#
# There is no nvcc-vs-clang split here: Metal code is .mm/.cpp compiled by the
# same clang++ clangd uses, and .metal shaders are built by `xcrun metal` (not
# by clangd). So the regular build's compile_commands.json is already fully
# clangd-compatible -- no separate `make dx` / build/clangd needed, and none of
# the CUDA .cu/.cuh handling from .clangd.cuda applies.
CompileFlags:
  CompilationDatabase: build
