{ pkgs ? import <nixpkgs> { config.allowUnfree = true; } }:

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
    cudaPackages_12.nsight_compute
    
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
