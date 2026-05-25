#!/bin/bash
set -e

cd vendor/metal-cpp
./SingleHeader/MakeSingleHeader.py Foundation/Foundation.hpp QuartzCore/QuartzCore.hpp Metal/Metal.hpp MetalFX/MetalFX.hpp
mv ./SingleHeader/Metal.hpp ../../include/metal/Metal.hpp
