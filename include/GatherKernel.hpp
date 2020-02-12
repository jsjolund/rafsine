#pragma once

#include "CudaMathHelper.h"
#include "CudaUtils.hpp"
#include "DdQq.hpp"

__global__ void GatherKernel(int* map,
                             int size,
                             int* stencil,
                             real* input,
                             real* output);
