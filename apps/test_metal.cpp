#include <util/metal.hpp>
#include "Foundation/NSString.hpp"
#include "Metal/MTLCommandQueue.hpp"
#include "Metal/MTLLibrary.hpp"
#include "Metal/MTLResource.hpp"
#include <fmt/format.h>

int main(int argc, char* argv[]) {
    fmt::println("Hello world");
    auto* pool = NS::AutoreleasePool::alloc()->init();

    MTL::Device* device = MTL::CreateSystemDefaultDevice();
    fmt::println("Metal works! {}, {}", device->name()->utf8String(), device->maxThreadgroupMemoryLength());

    NS::Error* error;
    NS::String* filePath = NS::String::string("./build/apps/default.metallib", NS::UTF8StringEncoding);
    MTL::Library* library = device->newLibrary(filePath, &error);
    //MTL::Library* library = device->newDefaultLibrary();

    if (error) {
        fmt::println("Error loading library: {}", error->localizedDescription()->utf8String());
        exit(1);
    }

    fmt::println("Library loaded");

    auto fn = library->newFunction(NS::String::string("fill_f32", NS::UTF8StringEncoding));
    auto pso = device->newComputePipelineState(fn, &error);

    if (error) {
        fmt::println("Error creating compute pipeline: {}", error->localizedDescription()->utf8String());
        exit(1);
    }

    MTL::CommandQueue* queue = device->newCommandQueue();

    auto cmd = queue->commandBuffer();
    auto enc = cmd->computeCommandEncoder();

    // 6 bf16 values
    auto* out_buf = device->newBuffer(6 * 4, MTL::ResourceStorageModeShared);
    float value = 4;
    uint n = 6;

    enc->setComputePipelineState(pso);
    enc->setBuffer(out_buf, 0, 0);
    enc->setBytes(&value, sizeof(float), 1);
    enc->setBytes(&n, sizeof(uint), 2);

    MTL::Size grid(n, 1, 1);

    MTL::Size tgs(std::min<uint>(pso->maxTotalThreadsPerThreadgroup(), 256), 1, 1);
    enc->dispatchThreads(grid, tgs);   // handles non-multiple grids
    enc->endEncoding();
    cmd->commit();
    cmd->waitUntilCompleted();

    auto* contents_ = reinterpret_cast<float*>(out_buf->contents());
    for(int i = 0; i < 6; ++i) {
        fmt::println("{}: {}", i, contents_[i]);
    }

    pool->release();
}
