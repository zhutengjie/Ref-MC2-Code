
#pragma once
#include "accessor.h"

struct BilateralDenoiserParams
{
    PackedTensorAccessor32<float, 4> col;
    PackedTensorAccessor32<float, 4> col_grad;  
    PackedTensorAccessor32<float, 4> nrm;
    PackedTensorAccessor32<float, 4> zdz;
    PackedTensorAccessor32<float, 4> out;
    PackedTensorAccessor32<float, 4> out_grad;
    float sigma;
};
