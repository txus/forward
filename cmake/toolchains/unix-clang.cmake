# Base toolchain: any Unix with modern LLVM Clang + libc++.
#
# Contract: Clang >= 20, libc++. The VERSION is enforced in CMakeLists.txt
# after project() -- CMAKE_CXX_COMPILER_VERSION is not known this early in a
# toolchain file.
#
# This file is include()d by the leaf toolchains (macos-metal.cmake,
# linux-cuda.cmake) AFTER they set the compiler path. It holds only what is
# common to every platform.

# Use libc++ explicitly. Both macOS (Homebrew LLVM) and NixOS deliberately use
# libc++; making it explicit keeps behaviour identical across platforms.
string(APPEND CMAKE_CXX_FLAGS_INIT " -stdlib=libc++")

# Note: CMAKE_CXX_EXTENSIONS (gnu++26) is set once in CMakeLists.txt, not here,
# to avoid a toolchain-vs-directory-scope precedence trap.
