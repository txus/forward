# macOS clangd config. The build's compile_commands.json is fully
# clangd-compatible: -isysroot is baked in by cmake/toolchains/macos-metal.cmake,
# so no flag surgery is needed here.
CompileFlags:
  CompilationDatabase: build
