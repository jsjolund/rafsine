#pragma once

#include <cuda.h>
#include <cuda_runtime_api.h>

#include "CudaUtils.hpp"
#include "DistributionArray.hpp"

__global__ void TestKernel(Partition partition, real *__restrict__ df,
                           int offset);
void runTestKernel(DistributionArray<real> *df, Partition partition,
                   int offset, cudaStream_t stream = 0);
// __global__ void TestBoundaryKernel(Partition partition, real *__restrict__
// df,
//                                    int offset);
// void runBoundaryTestKernel(DistributionArray<real> *df, Partition
// partition,
//                            int offset, cudaStream_t stream = 0);
