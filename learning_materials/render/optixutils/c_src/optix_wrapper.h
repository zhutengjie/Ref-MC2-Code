

#pragma once

#include <optix.h>
#include <string>
#include "envsampling/params.h"


//------------------------------------------------------------------------
// Python OptiX state wrapper.

struct OptiXState
{
    OptixDeviceContext context;
    OptixTraversableHandle gas_handle;
    CUdeviceptr            d_gas_output_buffer;

    // Differentiable env sampling
    OptixPipeline pipelineEnvSampling;
    OptixShaderBindingTable sbtEnvSampling;
    OptixModule moduleEnvSampling;

    OptixProgramGroup raygen_prog_group              = nullptr;
    OptixProgramGroup closesthit_prog_group          = nullptr;
    OptixProgramGroup miss_prog_group                = nullptr;
    OptixProgramGroup closesthit_occlusion_group     = nullptr;
    OptixProgramGroup miss_occlusion_group           = nullptr;
};


class OptiXStateWrapper
{
public:
    OptiXStateWrapper     (const std::string &path, const std::string &cuda_path);
    ~OptiXStateWrapper    (void);
    
    OptiXState*           pState;
};

void create_sbt(OptiXState * state, CUdeviceptr normals, CUdeviceptr kd, CUdeviceptr ks);
