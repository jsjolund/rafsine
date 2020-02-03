#pragma once

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <omp.h>

#include <iostream>

#include "gtest/gtest.h"

#include "CudaUtils.hpp"

// minimal test - 1 key per input index
struct test_xform {
  int m_nbins;
  float m_min, m_max;
  __host__ __device__ void operator()(float* input,
                                      int i,
                                      int* res_idx,
                                      int* res,
                                      int nres) const {
    float r = (input[i] - m_min) / (m_max - m_min) * (m_nbins - 1) + 0.5;
    *res_idx++ = static_cast<int>(r);
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
