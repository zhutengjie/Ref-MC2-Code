#include "../accessor.h"

struct EnvSamplingParams
{
    // Ray data
    PackedTensorAccessor32<float, 4>    ro;             // ray origin
    
    // GBuffer
    PackedTensorAccessor32<float, 3>    mask;
    PackedTensorAccessor32<float, 4>    gb_pos;
    PackedTensorAccessor32<float, 4>    gb_pos_grad;
    PackedTensorAccessor32<float, 4>    gb_normal;
    PackedTensorAccessor32<float, 4>    gb_normal_grad;
    PackedTensorAccessor32<float, 4>    gb_view_pos;
    PackedTensorAccessor32<float, 4>    gb_kd;
    PackedTensorAccessor32<float, 4>    gb_kd_grad;
    PackedTensorAccessor32<float, 4>    gb_ks;
    PackedTensorAccessor32<float, 4>    gb_ks_grad;
    
    // Light
    PackedTensorAccessor32<float, 3>    light;
    PackedTensorAccessor32<float, 3>    light_grad;
    PackedTensorAccessor32<float, 2>    pdf;        // light pdf
    PackedTensorAccessor32<float, 1>    rows;       // light sampling cdf
    PackedTensorAccessor32<float, 2>    cols;       // light sampling cdf

    // Output
    PackedTensorAccessor32<float, 4>    diff;
    PackedTensorAccessor32<float, 4>    diff_grad;
    PackedTensorAccessor32<float, 4>    spec;
    PackedTensorAccessor32<float, 4>    spec_grad;

    // Table with random permutations for stratified sampling
    PackedTensorAccessor32<int, 2>      perms;

    OptixTraversableHandle              handle;
    unsigned int                        BSDF;
    unsigned int                        n_samples_x;
    unsigned int                        rnd_seed;
    unsigned int                        backward;
    float                               shadow_scale;
    int                                 depth;

};


struct RadiancePRD
{
    // TODO: move some state directly into payload registers?
    unsigned int V;
    float3 shaded_color;
    int  depth;
};

struct RayGenData
{
};


struct MissData
{
    float4 bg_color;
};


struct HitGroupData
{
    float* normals;
    float* kd;
    float* ks;
};


template <typename T>
struct Record
{
    __align__( OPTIX_SBT_RECORD_ALIGNMENT ) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
    T data;
};

typedef Record<RayGenData>   RayGenRecord;
typedef Record<MissData>     MissRecord;
typedef Record<HitGroupData> HitGroupRecord;





