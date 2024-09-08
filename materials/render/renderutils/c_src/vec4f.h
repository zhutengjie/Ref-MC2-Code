

#pragma once 

struct vec4f
{
    float x, y, z, w;

#ifdef __CUDACC__
    __device__ vec4f() { }
    __device__ vec4f(float v) { x = v; y = v; z = v; w = v; }
    __device__ vec4f(float _x, float _y, float _z, float _w) { x = _x; y = _y; z = _z; w = _w; }
    __device__ vec4f(float4 v) { x = v.x; y = v.y; z = v.z; w = v.w; }
#endif
};

