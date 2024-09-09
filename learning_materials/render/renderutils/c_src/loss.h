

#pragma once

#include "common.h"

enum TonemapperType
{
    TONEMAPPER_NONE = 0,
    TONEMAPPER_LOG_SRGB = 1
};

enum LossType
{
    LOSS_L1 = 0,
    LOSS_MSE = 1,
    LOSS_RELMSE = 2,
    LOSS_SMAPE = 3,
    LOSS_N2N   = 4
};

struct LossKernelParams
{
    Tensor          img;
    Tensor          target;
    Tensor          out;
    dim3            gridSize;
    TonemapperType  tonemapper;
    LossType        loss;
};
