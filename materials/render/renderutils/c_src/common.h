

#pragma once
#include <cuda.h>
#include <stdint.h>

#include "vec3f.h"
#include "vec4f.h"
#include "tensor.h"

dim3 getLaunchBlockSize(int maxWidth, int maxHeight, dim3 dims);
dim3 getLaunchGridSize(dim3 blockSize, dim3 dims);

#ifdef __CUDACC__

#ifdef _MSC_VER
#define M_PI 3.14159265358979323846f
#endif

__host__ __device__ static inline dim3 getWarpSize(dim3 blockSize)
{
    return dim3(
        min(blockSize.x, 32u),
        min(max(32u / blockSize.x, 1u), min(32u, blockSize.y)),
        min(max(32u / (blockSize.x * blockSize.y), 1u), min(32u, blockSize.z))
    );
}

__device__ static inline float clamp(float val, float mn, float mx) { return min(max(val, mn), mx); }
#else
dim3 getWarpSize(dim3 blockSize);
#endif