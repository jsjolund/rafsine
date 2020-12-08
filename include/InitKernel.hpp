#pragma once

#include "CudaMathHelper.h"
#include "CudaUtils.hpp"
#include "DdQq.hpp"

template <int Q>
__global__ void InitKernel(real_t* __restrict__ df,
                           int nx,
                           int ny,
                           int nz,
                           float rho,
                           float vx,
                           float vy,
                           float vz,
                           float T);
