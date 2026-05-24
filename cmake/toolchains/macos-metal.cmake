# macOS / Metal toolchain. Requires Homebrew LLVM (AppleClang cannot compile
# C++26). Layers on top of unix-clang.cmake.

# 1. Locate Homebrew LLVM 20.
if(DEFINED ENV{HOMEBREW_PREFIX})
  set(_llvm "$ENV{HOMEBREW_PREFIX}/opt/llvm@20")
else()
  set(_llvm "/opt/homebrew/opt/llvm@20")
endif()

if(NOT EXISTS "${_llvm}/bin/clang++")
  message(FATAL_ERROR
    "Homebrew LLVM 20 not found at ${_llvm}/bin/clang++.\n"
    "Install it with:  brew install llvm@20")
endif()

set(CMAKE_C_COMPILER   "${_llvm}/bin/clang")
set(CMAKE_CXX_COMPILER "${_llvm}/bin/clang++")

# 2. Pin ONE macOS SDK so the compiler, find_library(), and clangd/clang-tidy
#    all agree. Without this, compile_commands.json carries no -isysroot and the
#    editor guesses a different SDK than the build -> phantom header errors
#    (mbstate_t, elaborated-enum-base) in files that compile fine.
#    We deliberately pin the Command Line Tools SDK (not `xcrun --show-sdk-path`
#    which returns the Xcode SDK): the CLT SDK is the one that parses cleanly
#    here. Fail with an actionable message if it is missing rather than letting
#    the build die later with a cryptic "sysroot not found".
set(CMAKE_OSX_SYSROOT "/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk")
if(NOT EXISTS "${CMAKE_OSX_SYSROOT}")
  message(FATAL_ERROR
    "macOS SDK not found at ${CMAKE_OSX_SYSROOT}.\n"
    "Install the Command Line Tools with:  xcode-select --install")
endif()

# 3. Shared base settings (libc++, C++ extensions).
include("${CMAKE_CURRENT_LIST_DIR}/unix-clang.cmake")
