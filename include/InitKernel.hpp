#pragma once

#include "CudaMathHelper.h"
#include "CudaUtils.hpp"
#include "DdQq.hpp"

__global__ void InitKernel(real* __restrict__ df,
                           real* __restrict__ dfT,
                           int nx,
                           int ny,
                           int nz,
                           float rho,
                           float vx,
                           float vy,
                           float vz,
                           float T,
                           float sq_term);
