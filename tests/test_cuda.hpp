#pragma once

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <omp.h>

#include <iostream>

#include "gtest/gtest.h"

#include "CudaUtils.hpp"

namespace cudatest {

class CudaTest : public testing::Test {
 public:
  CudaTest() {}

  ~CudaTest() {}

  void SetUp() {}

  void TearDown() {
    int numDevices;
    CUDA_RT_CALL(cudaGetDeviceCount(&numDevices));

#pragma omp parallel num_threads(numDevices)
    {
      const int srcDev = omp_get_thread_num();
      CUDA_RT_CALL(cudaSetDevice(srcDev));
      CUDA_RT_CALL(cudaFree(0));
      CUDA_RT_CALL(cudaDeviceReset());
    }
  }
};

}  // namespace cudatest
