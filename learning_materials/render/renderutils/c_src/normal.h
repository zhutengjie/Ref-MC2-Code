

#pragma once

#include "common.h"

struct PrepareShadingNormalKernelParams
{
    Tensor  pos;
    Tensor  view_pos;
    Tensor  perturbed_nrm;
    Tensor  smooth_nrm;
    Tensor  smooth_tng;
    Tensor  geom_nrm;
    Tensor  out;
    dim3    gridSize;
    bool    two_sided_shading, opengl;
};
