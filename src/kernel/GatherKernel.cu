#include "GatherKernel.hpp"

__global__ void GatherKernel(int* map,
                             int size,
                             int* stencil,
                             real_t* input,
                             real_t* output) {
  unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= size) return;
  if (stencil[i]) output[i] = input[map[i]];
}
