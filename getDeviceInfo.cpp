#include <iostream>
#include <hip/hip_runtime.h>

int main() {
    int deviceCount = 0;
    hipGetDeviceCount(&deviceCount);

    if (deviceCount == 0) {
        std::cout << "No HIP devices found." << std::endl;
        return 0;
    }

    for (int i = 0; i < deviceCount; ++i) {
        hipDeviceProp_t deviceProp;
        hipGetDeviceProperties(&deviceProp, i);

        std::cout << "Device " << i << ": " << deviceProp.name << std::endl;
        std::cout << "  Total Global Memory: " << deviceProp.totalGlobalMem << " bytes" << std::endl;
        std::cout << "  Shared Memory per Block: " << deviceProp.sharedMemPerBlock << " bytes" << std::endl;
        std::cout << "  Max Threads per Block: " << deviceProp.maxThreadsPerBlock << std::endl;
        std::cout << "  Max Threads Dimension: [" << deviceProp.maxThreadsDim[0] << ", "
                  << deviceProp.maxThreadsDim[1] << ", " << deviceProp.maxThreadsDim[2] << "]" << std::endl;
        std::cout << "  Max Grid Size: [" << deviceProp.maxGridSize[0] << ", "
                  << deviceProp.maxGridSize[1] << ", " << deviceProp.maxGridSize[2] << "]" << std::endl;
        std::cout << "  Max Threads Per MultiProcessor: " << deviceProp.maxThreadsPerMultiProcessor << std::endl;
        std::cout << "  Register per block: " << deviceProp.regsPerBlock << std::endl;
        // std::cout << "  Max VGPRs per CU: " << deviceProp.maxVgprsPerCU << std::endl;
        // std::cout << "Total SGPRs: " << deviceProp.maxRegistersPerBlock << std::endl;
    }
    return 0;
}