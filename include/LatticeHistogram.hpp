#pragma once

#include <cuda.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include "CudaUtils.hpp"

struct LatticeHistogramXform {
  int m_numBins;
  real_t m_min, m_max;

  LatticeHistogramXform() : m_numBins(1), m_min(0), m_max(0) {}

  __host__ __device__ void operator()(real_t* input,
                                      int i,
                                      int* res_idx,
                                      int* res,
                                      int nres) const;
};

// Sum-functor to be used for reduction - just a normal sum of two integers
struct LatticeHistogramSumFun {
  __device__ __host__ int operator()(int res1, int res2) const;
};

class LatticeHistogram {
 private:
  LatticeHistogramXform m_xform;
  LatticeHistogramSumFun m_sum;

 public:
  void calculate(thrust::device_vector<real_t>* src,
                 real_t min,
                 real_t max,
                 int numBins,
                 thrust::host_vector<int>* result,
                 const cudaStream_t stream = 0);
};
