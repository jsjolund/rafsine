#ifndef INCLUDE_KERNEL_HPP_
#define INCLUDE_KERNEL_HPP_

#include "BoundaryCondition.hpp"
#include "CFDScene.hpp"
#include "CudaMathHelper.h"
#include "CudaUtils.hpp"
#include "DdQq.hpp"

__global__ void InitKernel(real *__restrict__ df, real *__restrict__ dfT,
                           int nx, int ny, int nz, float rho, float vx,
                           float vy, float vz, float T, float sq_term);

__global__ void ComputeKernel(
    // Velocity distribution functions
    real *__restrict__ df, real *__restrict__ df_tmp,
    // Temperature distribution functions
    real *__restrict__ dfT, real *__restrict__ dfT_tmp,
    // Plot array for display
    real *__restrict__ plot,
    // Voxel type array
    const int *__restrict__ voxels,
    // Minimum of partition in global coordinates
    const glm::ivec3 partMin,
    // Maximum of partition in global coordinates
    const glm::ivec3 partMax,
    // Full size of the lattice
    const glm::ivec3 latticeSize,
    // Viscosity
    const real nu,
    // Smagorinsky constant
    const real C,
    // Thermal diffusivity
    const real nuT,
    // Turbulent Prandtl number
    const real Pr_t,
    // Gravity times thermal expansion
    const real gBetta,
    // Reference temperature for Boussinesq
    const real Tref,
    // Wuantity to be visualised
    const DisplayQuantity::Enum vis_q,
    // Contain the macroscopic temperature, velocity (x,y,z components)
    //  integrated in time (so /nbr_of_time_steps to get average)
    real *__restrict__ average,
    // Boundary condition data
    BoundaryCondition *__restrict__ bcs);

__global__ void HaloExchangeKernel(real *srcDfPtr, int2 **srcIdxPtrs,
                                   int srcQStride, real **dstDfPtrs,
                                   int2 **dstIdxPtrs, int *dstQStrides, int nq,
                                   int nNeighbours, int *idxLengths);

#endif  // INCLUDE_KERNEL_HPP_"
