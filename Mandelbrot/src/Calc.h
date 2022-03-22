#pragma once

#include <cstdint>
#include "cuda_kernel.h"

struct Vec2d
{
    Vec2d(long double x, long double y)
        : x(x)
        , y(y)
    {}
    Vec2d(const Vec2d& v)
        : x(v.x)
        , y(v.y)
    {}

    long double x, y;
};


class MandelbrotCalc
{
public:

    MandelbrotCalc();
    ~MandelbrotCalc();

    bool Updated();
    void AllocateBuffer(int h, int w);

    void ChangeXAxis(float input, int direction = 1);
    void ChangeYAxis(float input, int direction = 1);

    void ShiftXAxis(int direction);
    void ShiftYAxis(int direction);

    void IncreaseIterations();
    void DecreaseIterations();

    void ChangeColor(int coloring);

    void ChangeSet();

    void Calculate();
    void* GetBuffer();
    const size_t BufferSize() const { return height * width * sizeof(uint32_t); }

private:

    void CalcIncrements();

    int num_iterations = 100;

    bool updated{ true };

    int height;
    int width;

    Vec2d xScale;
    Vec2d yScale;

    long double xInc;
    long double yInc;

    int currentSet{ 1 };

    uint32_t* cudaBuffer;
    Kernel mKernel;


};
