
#pragma once

#include "common.h"

struct XfmKernelParams
{
    bool            isPoints;
    Tensor          points;
    Tensor          matrix;
    Tensor          out;
    dim3            gridSize;
};
