#define main search17_program_main
#include "search17_unroll16.cu"
#undef main

int main() {
    check_cuda(cudaSetDevice(0), "set device");
    cudaDeviceProp properties;
    check_cuda(cudaGetDeviceProperties(&properties, 0), "properties");
    std::cout << "sms=" << properties.multiProcessorCount
              << " max_threads_per_sm=" << properties.maxThreadsPerMultiProcessor
              << " registers_per_sm=" << properties.regsPerMultiprocessor << "\n";
    for (int block : {128, 256, 384, 512, 640, 768, 1024}) {
        int blocks = 0;
        check_cuda(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks, scan_kernel<true>, block, 0), "occupancy");
        std::cout << "block=" << block << " resident_blocks=" << blocks
                  << " warps=" << blocks * block / 32 << "\n";
    }
}
