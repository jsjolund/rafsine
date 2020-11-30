#pragma once

#include "BoundaryCondition.hpp"
#include "CFDScene.hpp"
#include "CudaMathHelper.h"
#include "CudaUtils.hpp"
#include "DdQq.hpp"
#include "LBM_BGK.hpp"
#include "LBM_MRT.hpp"
#include "PhysicalQuantity.hpp"

template <LBM::Enum method, D3Q4::Enum axis>
__global__ void ComputeKernel(
    const Partition partition,
    // Velocity distribution functions
    real_t* __restrict__ df,
    real_t* __restrict__ df_tmp,
    // Temperature distribution functions
    real_t* __restrict__ dfT,
    real_t* __restrict__ dfT_tmp,
    // Internal temperature distribution functions
    real_t* __restrict__ dfTeff,
    real_t* __restrict__ dfTeff_tmp,
    // Voxel type array
    const voxel_t* __restrict__ voxels,
    // Boundary condition data
    voxel_t* __restrict__ bcsId,
    VoxelType::Enum* __restrict__ bcsType,
    real_t* __restrict__ bcsTemperature,
    real3_t* __restrict__ bcsVelocity,
    int3* __restrict__ bcsNormal,
    int3* __restrict__ bcsRelPos,
    real_t* __restrict__ bcsTau1,
    real_t* __restrict__ bcsTau2,
    real_t* __restrict__ bcsLambda,
    // Time step in seconds
    const real_t dt,
    // Viscosity
    const real_t nu,
    // Smagorinsky constant
    const real_t C,
    // Thermal diffusivity
    const real_t nuT,
    // Turbulent Prandtl number
    const real_t Pr_t,
    // Gravity times thermal expansion
    const real_t gBetta,
    // Reference temperature for Boussinesq
    const real_t Tref,
    // Contain the macroscopic temperature, velocity (x,y,z components)
    //  integrated in time (so /nbr_of_time_steps to get average)
    real_t* __restrict__ averageSrc,
    real_t* __restrict__ averageDst,
    const DisplayQuantity::Enum displayQuantity,
    real_t* __restrict__ plot);
