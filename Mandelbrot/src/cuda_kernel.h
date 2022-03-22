#pragma once

#include <cstdint>


class Kernel
{
public:

	Kernel();
	~Kernel();

	// Prepare Kernel
	void KernelAllocate(int w, int h);

	// Kernel Calls
	void KernelCall(const double& xStart, const double& xInc, const double& yStart, const double& yInc);
	void CalculateN(const double& xStart, const double& xInc, const double& yStart, const double& yInc);
	void CalculateColor();

	// Get Color Buffer
	uint32_t* GetColorBuffer();

	// Get Set
	int GetSet();

	// Setters
	void SetIterations(int iterations);
	void SetColoring(int method);
	void SetCalculationSet(int set);

private:

	int* itBuffer;
	uint32_t* colorBuffer;
	int calc_set{ 1 };
	int width, height;
	int max_iterations;
	int color_method{ 1 };

};
