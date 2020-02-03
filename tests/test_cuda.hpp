#pragma once

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <omp.h>

#include <iostream>

#include "gtest/gtest.h"

#include "CudaUtils.hpp"

// minimal test - 1 key per input index
struct test_xform {
  __host__ __device__ void operator()(int* input,
                                      int i,
                                      int* res_idx,
                                      int* res,
                                      int nres) const {
    *res_idx++ = input[i];
    *res++ = 1;
  }
};

// Sum-functor to be used for reduction - just a normal sum of two integers
struct test_sumfun {
  __device__ __host__ int operator()(int res1, int res2) const {
    return res1 + res2;
  }
};

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
