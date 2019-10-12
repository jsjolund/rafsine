#include "CudaUtils.hpp"

__device__ int idx1d(void) {
  return blockIdx.y * (gridDim.x * blockDim.x) + blockDim.x * blockIdx.x +
         threadIdx.x;
}
