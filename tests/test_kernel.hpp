#pragma once

#include <cuda.h>
#include <cuda_runtime_api.h>

#include "CudaUtils.hpp"
#include "DistributionArray.hpp"

__global__ void TestKernel(real *__restrict__ df, glm::ivec3 pMin,
                           glm::ivec3 pMax, glm::ivec3 pHalo, int scl);

void runTestKernel(DistributionArray *df, SubLattice subLattice, int scl,
                   cudaStream_t stream = 0);
