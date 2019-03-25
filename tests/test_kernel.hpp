#pragma once

#include <cuda.h>
#include <cuda_runtime_api.h>

#include "CudaUtils.hpp"
#include "DistributionArray.hpp"

__global__ void TestKernel(SubLattice subLattice, real *__restrict__ df,
                           int offset);
void runTestKernel(DistributionArray<real> *df, SubLattice subLattice,
                   int offset, cudaStream_t stream = 0);
// __global__ void TestBoundaryKernel(SubLattice subLattice, real *__restrict__
// df,
//                                    int offset);
// void runBoundaryTestKernel(DistributionArray<real> *df, SubLattice
// subLattice,
//                            int offset, cudaStream_t stream = 0);
