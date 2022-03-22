#include "cuda_kernel.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cstdlib>
#include <iostream>


#define checkCudaErrors(val) check_cuda( (val), #val, __FILE__, __LINE__ )
void check_cuda(cudaError_t result, char const* const func, const char* const file, int const line) {
    if (result) {
        std::cerr << "CUDA error = " << static_cast<unsigned int>(result) << " at " <<
            file << ":" << line << " '" << func << "' \n";
        // Make sure we call CUDA Device Reset before exiting
        cudaDeviceReset();
        exit(99);
    }
}


__global__ void render(uint32_t* buffer, int width, int height, int max_iterations, double xStart, double xInc, double yStart, double yInc)
{
    int w = threadIdx.x + blockIdx.x * blockDim.x;
    int h = threadIdx.y + blockIdx.y * blockDim.y;
    if ((w >= width) || (h >= height)) return;
    int pixel_index = h * width + w;


    // find n
    int n = 0;
    double cx, cy;

    cx = xStart + w * xInc;
    cy = yStart + h * yInc;

    double x2 = 0;
    double y2 = 0;
    double ww = 0;

    while ((x2 + y2) <= 4.0 && n < max_iterations)
    {
        double x = x2 - y2 + cx;
        double y = ww - x2 - y2 + cy;
        x2 = x * x;
        y2 = y * y;
        ww = (x + y) * (x + y);
        n++;
    }


    // color
    if (n == max_iterations)
        buffer[pixel_index] = 0x00ffffff;
    else if (n == 1)
        buffer[pixel_index] = 0x00000000;
    else
    {
        uint8_t red = static_cast<uint8_t>(n);
        uint8_t green = static_cast<uint8_t>(0);
        uint8_t blue = static_cast<uint8_t>(n * 2);
        uint32_t color = 0xff << 24 | red << 16 | green << 8 | blue;
        buffer[pixel_index] = color;
    }

}





void kernel::kernel_init(int width, int height)
{
    auto buffer_size = sizeof(uint32_t) * height * width;
    checkCudaErrors(cudaMallocManaged((void**)&buffer_ptr, buffer_size));
}


void kernel::kernel(int width, int height, int max_iterations, double xStart, double xInc, double yStart, double yInc)
{
    int tx = 8;
    int ty = 8;

    dim3* blocks = nullptr;
    dim3* threads = nullptr;

    *blocks = dim3(width / tx + 1, height / ty + 1);
    *threads = dim3(tx, ty);

    render << < *blocks, *threads >> > (buffer_ptr, width, height, max_iterations, xStart, xInc, yStart, yInc);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
}