#pragma once

#include "BoundaryCondition.hpp"
#include "CFDScene.hpp"
#include "CudaMathHelper.h"
#include "CudaUtils.hpp"
#include "DdQq.hpp"
#include "PhysicalQuantity.hpp"

__device__ PhysicalQuantity computeBGK(const int x,
                                       const int y,
                                       const int z,
                                       const int nx,
                                       const int ny,
                                       const int nz,
                                       const real nu,
                                       const real nuT,
                                       const real C,
                                       const real Pr_t,
                                       const real gBetta,
                                       const real Tref,
                                       const real f0,
                                       const real f1,
                                       const real f2,
                                       const real f3,
                                       const real f4,
                                       const real f5,
                                       const real f6,
                                       const real f7,
                                       const real f8,
                                       const real f9,
                                       const real f10,
                                       const real f11,
                                       const real f12,
                                       const real f13,
                                       const real f14,
                                       const real f15,
                                       const real f16,
                                       const real f17,
                                       const real f18,
                                       const real T0,
                                       const real T1,
                                       const real T2,
                                       const real T3,
                                       const real T4,
                                       const real T5,
                                       const real T6,
                                       real* __restrict__ df_tmp,
                                       real* __restrict__ dfT_tmp,
                                       PhysicalQuantity* phy);
