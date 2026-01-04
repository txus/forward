{ pkgs ? import <nixpkgs> { config.allowUnfree = true; } }:

let
  # Fixed nsight_compute package with correct directory structure
  # The ncu binary at bin/target/linux-desktop-glibc_2_11_3-x64/ncu uses readlink -f on itself
  # and looks for ../../sections relative to its real path. In nixpkgs, sections are at pkg root,
  # not at bin/sections. Since readlink -f resolves symlinks, we must COPY the binary
  # to a new location where the relative path works.
  nsight_compute_fixed = pkgs.stdenv.mkDerivation {
    pname = "nsight-compute-fixed";
    version = pkgs.cudaPackages_12.nsight_compute.version;
    src = pkgs.cudaPackages_12.nsight_compute;
    dontUnpack = true;
    dontBuild = true;
    dontStrip = true;
    dontPatchELF = true;
    installPhase = ''
      mkdir -p $out/bin/target/linux-desktop-glibc_2_11_3-x64

      # The key fix: sections at bin/sections so ../../sections from bin/target/*/ncu works
      ln -s $src/sections $out/bin/sections

      # Copy the actual binary and its libs (readlink -f will now find our copy)
      cp -a $src/bin/target/linux-desktop-glibc_2_11_3-x64/* $out/bin/target/linux-desktop-glibc_2_11_3-x64/

      # Create wrapper script
      cat > $out/bin/ncu << 'WRAPPER'
#!/bin/sh
APPDIR="$(dirname "$(readlink -f -- "$0")")"
ARCH="$(uname -m)"
if [ "$ARCH" = "x86_64" ]; then
    "$APPDIR/target/linux-desktop-glibc_2_11_3-x64/ncu" "$@"
elif [ "$ARCH" = "aarch64" ]; then
    "$APPDIR/target/linux-desktop-t210-a64/ncu" "$@"
else
    echo "Unsupported Architecture: $ARCH"
fi
WRAPPER
      chmod +x $out/bin/ncu

      # Link other directories
      for d in $src/*; do
        name=$(basename "$d")
        if [ "$name" != "bin" ] && [ "$name" != "sections" ]; then
          ln -s "$d" $out/
        fi
      done
      # Also link sections at root for compatibility
      ln -s $src/sections $out/sections
    '';
  };
in
(pkgs.mkShell.override { stdenv = pkgs.clangStdenv; }) {
  nativeBuildInputs = with pkgs; [
    # Use clang from stdenv
    clang_20
    llvm_20
    lld_20

    # Build tools
    cmake
    ninja
    ccache
    gnumake
    cargo

    # Debugging & profiling
    lldb_20
    gdb
    valgrind
    perf

    # Code quality
    clang-tools
    cppcheck
    
    # CUDA
    cudaPackages_12.cudatoolkit
    cudaPackages_12.cudnn
    cudaPackages_12.nsight_systems
    nsight_compute_fixed  # Fixed package with correct sections symlink (replaces nsight_compute)
    
    # Utilities
    bear
    git
    ripgrep
    fd
  ];

  buildInputs = with pkgs; [
    cudaPackages_12.cuda_cudart
    cudaPackages_12.libcublas
    llvmPackages_20.openmp
    llvmPackages_20.libcxx  # Use libc++ to avoid __noinline__ macro conflict with libstdc++
  ];

  # Disable hardening flags incompatible with CUDA/NVPTX target
  # zerocallusedregs: -fzero-call-used-regs=used-gpr is unsupported for nvptx64
  # fortify: causes exception specification mismatches in CUDA
  hardeningDisable = [ "zerocallusedregs" "fortify" ];

  shellHook = ''
    export CUDA_PATH="${pkgs.cudaPackages_12.cudatoolkit}"
    export CUDACXX="${pkgs.cudaPackages_12.cudatoolkit}/bin/nvcc"
    export LD_LIBRARY_PATH="${pkgs.cudaPackages_12.cudatoolkit}/lib:${pkgs.cudaPackages_12.cudnn}/lib:${pkgs.llvmPackages_20.openmp}/lib:$LD_LIBRARY_PATH"

    # CUDA compiler flags
    export CXXFLAGS="-I${pkgs.cudaPackages_12.cudatoolkit}/include"
    export LDFLAGS="-L${pkgs.cudaPackages_12.cudatoolkit}/lib"

    # For CMake presets
    export CLANGXX_PATH="${pkgs.clang_20}/bin/clang++"
    export OPENMP_ROOT="${pkgs.llvmPackages_20.openmp}"
    export CUDA_TOOLKIT_ROOT="${pkgs.cudaPackages_12.cudatoolkit}"

    # Clang resource directory for clangd (NixOS-specific)
    export CLANG_RESOURCE_DIR="$(clang++ -print-resource-dir)"

    # C++ standard library include path for clangd CUDA files (query-driver fails for -x cuda)
    export LIBCXX_INCLUDE="${pkgs.llvmPackages_20.libcxx.dev}/include/c++/v1"

    # CPATH for clangd - includes C++ stdlib and glibc headers
    # clangStdenv uses GCC's libstdc++, so we need those paths
    export CPATH="${pkgs.gcc.cc}/include/c++/${pkgs.gcc.version}:${pkgs.gcc.cc}/include/c++/${pkgs.gcc.version}/x86_64-unknown-linux-gnu:${pkgs.glibc.dev}/include"

    echo "C++/CUDA development environment loaded!"
    echo "Clang version: $(clang --version | head -n1)"
    echo "CMake version: $(cmake --version | head -n1)"
    echo "CUDA path: $CUDA_PATH"
    echo "LLDB: $(lldb --version | head -n1)"
    echo "OpenMP: ${pkgs.llvmPackages_20.openmp}"
    echo ""
    echo "CUDA Profiling Tools:"
    echo "  - nsys: Nsight Systems (timeline profiler)"
    echo "  - ncu:  Nsight Compute (kernel profiler)"
    echo ""
    echo "Build configurations:"
    echo "  Debug:          cmake -B build -DCMAKE_BUILD_TYPE=Debug"
    echo "  RelWithDebInfo: cmake -B build -DCMAKE_BUILD_TYPE=RelWithDebInfo"
    echo "  Release:        cmake -B build -DCMAKE_BUILD_TYPE=Release"

    # Test stdlib.h is found
    echo ""
    echo '#include <stdlib.h>' | clang++ -x c++ -E - > /dev/null 2>&1 && echo "✓ stdlib.h found" || echo "✗ stdlib.h NOT found"
  '';
}
