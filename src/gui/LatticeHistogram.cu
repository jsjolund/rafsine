#include "LatticeHistogram.hpp"

#include "cuda_histogram.h"

__host__ __device__ void LatticeHistogramXform::operator()(real_t* input,
                                                           int i,
                                                           int* res_idx,
                                                           int* res,
                                                           int) const {
  real_t r = (input[i] - m_min) / (m_max - m_min) * (m_numBins - 1) + 0.5;
  *res_idx++ = static_cast<int>(r);
  *res++ = 1;
}

__device__ __host__ int LatticeHistogramSumFun::operator()(int res1,
                                                           int res2) const {
  return res1 + res2;
}

void LatticeHistogram::calculate(thrust::device_vector<real_t>* src,
                                 real_t min,
                                 real_t max,
                                 int numBins,
                                 thrust::host_vector<int>* result,
                                 const cudaStream_t stream) {
  m_xform.m_min = min;
  m_xform.m_max = max;
  m_xform.m_numBins = numBins;
  thrust::fill(result->begin(), result->end(), 0);
  real_t* srcPtr = thrust::raw_pointer_cast(&(*src)[0]);
  int* resultPtr = thrust::raw_pointer_cast(&(*result)[0]);
  callHistogramKernel<histogram_atomic_inc, 1>(
      srcPtr, m_xform, m_sum, 0, static_cast<int>(src->size()), 0, resultPtr,
      m_xform.m_numBins, false, stream);
}
