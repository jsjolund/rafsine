#pragma once

#include <cuda.h>
#include <cuda_runtime_api.h>

#include "CudaUtils.hpp"
#include "DistributionArray.hpp"

__global__ void TestKernel(SubLattice subLattice, real *__restrict__ df,
                           int scl);
__global__ void TestBoundaryKernel(SubLattice subLattice, real *__restrict__ df,
                                   int scl);
void runTestKernel(DistributionArray<real> *df, SubLattice subLattice, int scl,
                   cudaStream_t stream = 0);
void runBoundaryTestKernel(DistributionArray<real> *df, SubLattice subLattice,
                           int scl, cudaStream_t stream = 0);
