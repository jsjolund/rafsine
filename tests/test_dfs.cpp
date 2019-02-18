#include <gtest/gtest.h>

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <omp.h>

#include "CudaUtils.hpp"
#include "DistributionFunction.hpp"
#include "KernelInterface.hpp"
#include "test_kernel.hpp"

TEST(DistributedDFTest, HaloExchangeMultiGPU) {
  // int maxDevices = 2, nq = 19, nx = 5, ny = 3, nz = 2;
  int maxDevices = 2;

  int numDevices;
  CUDA_RT_CALL(cudaGetDeviceCount(&numDevices));
  numDevices = min(numDevices, maxDevices);
  CUDA_RT_CALL(cudaSetDevice(0));
}
