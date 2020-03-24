#pragma once

#include "BoundaryCondition.hpp"
#include "CFDScene.hpp"
#include "CudaMathHelper.h"
#include "CudaUtils.hpp"
#include "DdQq.hpp"
#include "LbmMethod.hpp"
#include "PhysicalQuantity.hpp"

#include "LBM-BGK.h"
#include "LBM-MRT.h"

template <LBM::Enum method, D3Q4::Enum axis>
__global__ void ComputeKernel(
    const Partition partition,
    // Velocity distribution functions
    real* __restrict__ df,
    real* __restrict__ df_tmp,
    // Temperature distribution functions
    real* __restrict__ dfT,
    real* __restrict__ dfT_tmp,
    // Internal temperature distribution functions
    real* __restrict__ dfTeff,
    real* __restrict__ dfTeff_tmp,
    // Voxel type array
    const voxel_t* __restrict__ voxels,
    // Boundary condition data
    BoundaryCondition* __restrict__ bcs,
    // Time step in seconds
    const real dt,
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
    // Contain the macroscopic temperature, velocity (x,y,z components)
    //  integrated in time (so /nbr_of_time_steps to get average)
    real* __restrict__ averageSrc,
    real* __restrict__ averageDst,
    const DisplayQuantity::Enum displayQuantity,
    real* __restrict__ plot);
