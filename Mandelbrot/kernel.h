#pragma once

#include <cstdint>



namespace kernel
{

	void kernel_init(uint32_t* buffer, int width, int height);
	void kernel(uint32_t* buffer, int width, int height, int max_iterations, double xStart, double xInc, double yStart, double yInc);

}
