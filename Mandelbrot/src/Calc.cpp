#include <complex>
#include <iostream>
#include "Calc.h"




constexpr int MAX_ITERATION = 100;


void MandelbrotCalc::ChangeColor(int coloring)
{
    if (coloring == 1)
        mKernel.SetColoring(1);
    if (coloring == -1)
        mKernel.SetColoring(2);
    updated = true;
}

void MandelbrotCalc::ChangeSet()
{
    if (mKernel.GetSet() == 1)
        mKernel.SetCalculationSet(-1);
    else
        mKernel.SetCalculationSet(1);
    updated = true;
}

void MandelbrotCalc::IncreaseIterations()
{
    num_iterations *= 10;
    mKernel.SetIterations(num_iterations);
    updated = true;
}

void MandelbrotCalc::DecreaseIterations()
{
    num_iterations *= 0.1;
    mKernel.SetIterations(num_iterations);
    updated = true;
}

bool MandelbrotCalc::Updated()
{
    return updated;
}

void* MandelbrotCalc::GetBuffer() { 
    return (void*)mKernel.GetColorBuffer();
};


void MandelbrotCalc::ShiftXAxis(int direction)
{
    double delta = (xScale.y - xScale.x) * 0.05;
    if (direction == 1)
    {
        xScale.x += delta;
        xScale.y += delta;
    }
    else if (direction == -1)
    {
        xScale.x -= delta;
        xScale.y -= delta;
    }
    updated = true;
}

void MandelbrotCalc::ShiftYAxis(int direction)
{
    double delta = (yScale.y - yScale.x) * 0.05;
    if (direction == 1)
    {
        yScale.x += delta;
        yScale.y += delta;
    }
    else if (direction == -1)
    {
        yScale.x -= delta;
        yScale.y -= delta;
    }
    updated = true;
}


void MandelbrotCalc::ChangeXAxis(float input, int direction)
{
    long double delta = (xScale.y - xScale.x) * 0.05;
    auto delta_left = input * delta;
    auto delta_right = delta - delta_left;
    if (direction == 1)
    {
        xScale.x += delta_left;
        xScale.y -= delta_right;
    }
    else if (direction == -1)
    {
        xScale.x -= delta_left;
        xScale.y += delta_right;
    }
    CalcIncrements();
    updated = true;
}

void MandelbrotCalc::ChangeYAxis(float input, int direction)
{
    double aspect_ratio = (float)height / width;
    long double delta = (yScale.y - yScale.x) * (0.05);
    auto delta_top = input * delta;
    auto delta_bot = delta - delta_top;
    if (direction == 1)
    {
        yScale.x += delta_top;
        yScale.y -= delta_bot;
    }
    else if (direction == -1)
    {
        yScale.x -= delta_top;
        yScale.y += delta_bot;
    }
    CalcIncrements();
    updated = true;
}

void MandelbrotCalc::CalcIncrements()
{
    xInc = abs(xScale.y - xScale.x) / width;
    yInc = abs(yScale.y - yScale.x) / height;
}

void MandelbrotCalc::AllocateBuffer(int h, int w)
{
    height = h;
    width = w;
    CalcIncrements();
    mKernel.KernelAllocate(w, h);
}



void MandelbrotCalc::Calculate()
{
    mKernel.KernelCall(xScale.x, xInc, yScale.x, yInc);
    updated = false;
}

MandelbrotCalc::MandelbrotCalc()
    : xScale(-2.0, 0.47)
    , yScale(-1.12, 1.12)
    , cudaBuffer(nullptr)
    , mKernel()
{}

MandelbrotCalc::~MandelbrotCalc()
{}
