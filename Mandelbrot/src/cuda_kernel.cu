#include "cuda_kernel.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cstdlib>
#include <iostream>
#include <thrust/complex.h>


typedef struct {
    double r;       // a fraction between 0 and 1
    double g;       // a fraction between 0 and 1
    double b;       // a fraction between 0 and 1
} rgb;

typedef struct {
    double h;       // angle in degrees
    double s;       // a fraction between 0 and 1
    double v;       // a fraction between 0 and 1
} hsv;

__device__ rgb hsv2rgb(hsv in)
{
    double      hh, p, q, t, ff;
    long        i;
    rgb         out;

    if (in.s <= 0.0) {       // < is bogus, just shuts up warnings
        out.r = in.v;
        out.g = in.v;
        out.b = in.v;
        return out;
    }
    hh = in.h;
    if (hh >= 360.0) hh = 0.0;
    hh /= 60.0;
    i = (long)hh;
    ff = hh - i;
    p = in.v * (1.0 - in.s);
    q = in.v * (1.0 - (in.s * ff));
    t = in.v * (1.0 - (in.s * (1.0 - ff)));

    switch (i) {
    case 0:
        out.r = in.v;
        out.g = t;
        out.b = p;
        break;
    case 1:
        out.r = q;
        out.g = in.v;
        out.b = p;
        break;
    case 2:
        out.r = p;
        out.g = in.v;
        out.b = t;
        break;

    case 3:
        out.r = p;
        out.g = q;
        out.b = in.v;
        break;
    case 4:
        out.r = t;
        out.g = p;
        out.b = in.v;
        break;
    case 5:
    default:
        out.r = in.v;
        out.g = p;
        out.b = q;
        break;
    }
    return out;
}

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


__global__ void getcolor(int* itBuffer, uint32_t* colorBuffer, int width, int height, int max_iterations, int color_method)
{
    int w = threadIdx.x + blockIdx.x * blockDim.x;
    int h = threadIdx.y + blockIdx.y * blockDim.y;
    if ((w >= width) || (h >= height)) return;
    int pixel_index = h * width + w;

    int& n = itBuffer[pixel_index];

    if (color_method == 1)
    {
        // coloring 1
        if (n == max_iterations)
            colorBuffer[pixel_index] = 0x00000000;
        else if (n == 1)
            colorBuffer[pixel_index] = 0x00000000;
        else
        {
            uint8_t red = static_cast<uint8_t>(n);
            uint8_t green = static_cast<uint8_t>(0);
            uint8_t blue = static_cast<uint8_t>(n * 2);
            uint32_t color = 0xff << 24 | red << 16 | green << 8 | blue;
            colorBuffer[pixel_index] = color;
        }

    }

    if (color_method == 2)
    {
        double red, green, blue;
        red = (0.5 * sin(n * 0.1) + 0.5) * 255.9;
        green = (0.5 * sin(n * 0.1 + 2.094) + 0.5f) * 255.9;
        blue = (0.5 * sin(n * 0.1 + 4.18) + 0.5f) * 255.9;

        uint32_t color = 0xff << 24 | static_cast<uint8_t>(red) << 16 | static_cast<uint8_t>(green) << 8 | static_cast<uint8_t>(blue);
        colorBuffer[pixel_index] = color;

    }


}

__global__ void getn(int* buffer, int width, int height, int max_iterations, double xStart, double xInc, double yStart, double yInc, int set)
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

    if (set == 1)
    {
        while ((x2 + y2) <= 4.0 && n < max_iterations)
        {
            double x = x2 - y2 + cx;
            double y = ww - x2 - y2 + cy;
            x2 = x * x;
            y2 = y * y;
            ww = (x + y) * (x + y);
            n++;
        }
    }
    else if (set == -1)
    {
        const double cr = -0.8;
        const double ci = 0.156;


        while (cx * cx + cy * cy < 4.0 && n < max_iterations)
        {
            double tmp = cx * cx - cy * cy;
            cy = 2 * cx * cy + ci;
            cx = tmp + cr;
            n++;
        }
    }

    buffer[pixel_index] = n;
}


Kernel::Kernel()
    : width(0)
    , height(0)
    , max_iterations(100)
{}

Kernel::~Kernel()
{
    checkCudaErrors(cudaFree(itBuffer));
    checkCudaErrors(cudaFree(colorBuffer));
}


void Kernel::SetCalculationSet(int set)
{
    calc_set = set;
}


int Kernel::GetSet() 
{ 
    return calc_set; 
}

void Kernel::KernelAllocate(int w, int h)
{
    width = w;
    height = h;
    auto buffer_size = height * width;
    checkCudaErrors(cudaMallocManaged((void**)&itBuffer, buffer_size * sizeof(int)));
    checkCudaErrors(cudaMallocManaged((void**)&colorBuffer, buffer_size * sizeof(uint32_t)));
}



uint32_t* Kernel::GetColorBuffer()
{
    return colorBuffer;
}

void Kernel::SetIterations(int iterations)
{
    max_iterations = iterations;
}

void Kernel::KernelCall(const double& xStart, const double& xInc, const double& yStart, const double& yInc)
{
    CalculateN(xStart, xInc, yStart, yInc);
    CalculateColor();
}

void Kernel::CalculateN(const double& xStart, const double& xInc, const double& yStart, const double& yInc)
{
    int tx = 8;
    int ty = 8;

    dim3 blocks(width / tx + 1, height / ty + 1);
    dim3 threads(tx, ty);

    getn << < blocks, threads >> > (itBuffer, width, height, max_iterations, xStart, xInc, yStart, yInc, calc_set);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
}

void Kernel::SetColoring(int method)
{
    color_method = method;
}

void Kernel::CalculateColor()
{
    int tx = 8;
    int ty = 8;

    dim3 blocks(width / tx + 1, height / ty + 1);
    dim3 threads(tx, ty);

    getcolor << < blocks, threads >> > (itBuffer, colorBuffer, width, height, max_iterations, color_method);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
}
