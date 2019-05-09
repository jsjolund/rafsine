#pragma once

#include "BoundaryCondition.hpp"
#include "CFDScene.hpp"
#include "CudaMathHelper.h"
#include "CudaUtils.hpp"
#include "DdQq.hpp"
#include "PhysicalQuantity.hpp"

__device__ PhysicalQuantity compute(
    // Lattice position in partition
    const glm::ivec3 pos,
    // Size of partition
    const glm::ivec3 size,
    // Size of halo
    const glm::ivec3 halo,
    // Velocity distribution functions
    real *__restrict__ df, real *__restrict__ df_tmp,
    // Temperature distribution functions
    real *__restrict__ dfT, real *__restrict__ dfT_tmp,
    // Voxel type array
    const int *__restrict__ voxels,
    // Boundary condition data
    BoundaryCondition *__restrict__ bcs,
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
    const real Tref);

__device__ void computeAndPlot(
    // Lattice position in partition
    const glm::ivec3 pos,
    // Size of partition
    const glm::ivec3 size,
    // Size of halo
    const glm::ivec3 halo,
    // Velocity distribution functions
    real *__restrict__ df, real *__restrict__ df_tmp,
    // Temperature distribution functions
    real *__restrict__ dfT, real *__restrict__ dfT_tmp,
    // Plot array for display
    real *__restrict__ plot,
    // Contain the macroscopic temperature, velocity (x,y,z components)
    //  integrated in time (so /nbr_of_time_steps to get average)
    real *__restrict__ averageSrc, real *__restrict__ averageDst,
    // Voxel type array
    const int *__restrict__ voxels,
    // Boundary condition data
    BoundaryCondition *__restrict__ bcs,
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
    // Quantity to be visualised
    const DisplayQuantity::Enum vis_q);

__global__ void ComputeKernel(
    const SubLattice subLattice, real *__restrict__ df,
    real *__restrict__ df_tmp, real *__restrict__ dfT,
    real *__restrict__ dfT_tmp, real *__restrict__ plot,
    real *__restrict__ averageSrc, real *__restrict__ averageDst,
    const int *__restrict__ voxels, BoundaryCondition *__restrict__ bcs,
    const real nu, const real C, const real nuT, const real Pr_t,
    const real gBetta, const real Tref, const DisplayQuantity::Enum vis_q);

__global__ void ComputeKernelInterior(
    const SubLattice subLattice, real *__restrict__ df,
    real *__restrict__ df_tmp, real *__restrict__ dfT,
    real *__restrict__ dfT_tmp, real *__restrict__ plot,
    real *__restrict__ averageSrc, real *__restrict__ averageDst,
    const int *__restrict__ voxels, BoundaryCondition *__restrict__ bcs,
    const real nu, const real C, const real nuT, const real Pr_t,
    const real gBetta, const real Tref, const DisplayQuantity::Enum vis_q);

__global__ void ComputeKernelBoundaryX(
    const SubLattice subLattice, real *__restrict__ df,
    real *__restrict__ df_tmp, real *__restrict__ dfT,
    real *__restrict__ dfT_tmp, real *__restrict__ plot,
    real *__restrict__ averageSrc, real *__restrict__ averageDst,
    const int *__restrict__ voxels, BoundaryCondition *__restrict__ bcs,
    const real nu, const real C, const real nuT, const real Pr_t,
    const real gBetta, const real Tref, const DisplayQuantity::Enum vis_q);

__global__ void ComputeKernelBoundaryY(
    const SubLattice subLattice, real *__restrict__ df,
    real *__restrict__ df_tmp, real *__restrict__ dfT,
    real *__restrict__ dfT_tmp, real *__restrict__ plot,
    real *__restrict__ averageSrc, real *__restrict__ averageDst,
    const int *__restrict__ voxels, BoundaryCondition *__restrict__ bcs,
    const real nu, const real C, const real nuT, const real Pr_t,
    const real gBetta, const real Tref, const DisplayQuantity::Enum vis_q);

__global__ void ComputeKernelBoundaryZ(
    const SubLattice subLattice, real *__restrict__ df,
    real *__restrict__ df_tmp, real *__restrict__ dfT,
    real *__restrict__ dfT_tmp, real *__restrict__ plot,
    real *__restrict__ averageSrc, real *__restrict__ averageDst,
    const int *__restrict__ voxels, BoundaryCondition *__restrict__ bcs,
    const real nu, const real C, const real nuT, const real Pr_t,
    const real gBetta, const real Tref, const DisplayQuantity::Enum vis_q);

__global__ void ComputeAndPlotKernel(
    const SubLattice subLattice, real *__restrict__ df,
    real *__restrict__ df_tmp, real *__restrict__ dfT,
    real *__restrict__ dfT_tmp, real *__restrict__ plot,
    real *__restrict__ averageSrc, real *__restrict__ averageDst,
    const int *__restrict__ voxels, BoundaryCondition *__restrict__ bcs,
    const real nu, const real C, const real nuT, const real Pr_t,
    const real gBetta, const real Tref, const DisplayQuantity::Enum vis_q);

__global__ void ComputeAndPlotKernelInterior(
    const SubLattice subLattice, real *__restrict__ df,
    real *__restrict__ df_tmp, real *__restrict__ dfT,
    real *__restrict__ dfT_tmp, real *__restrict__ plot,
    real *__restrict__ averageSrc, real *__restrict__ averageDst,
    const int *__restrict__ voxels, BoundaryCondition *__restrict__ bcs,
    const real nu, const real C, const real nuT, const real Pr_t,
    const real gBetta, const real Tref, const DisplayQuantity::Enum vis_q);

__global__ void ComputeAndPlotKernelBoundaryX(
    const SubLattice subLattice, real *__restrict__ df,
    real *__restrict__ df_tmp, real *__restrict__ dfT,
    real *__restrict__ dfT_tmp, real *__restrict__ plot,
    real *__restrict__ averageSrc, real *__restrict__ averageDst,
    const int *__restrict__ voxels, BoundaryCondition *__restrict__ bcs,
    const real nu, const real C, const real nuT, const real Pr_t,
    const real gBetta, const real Tref, const DisplayQuantity::Enum vis_q);

__global__ void ComputeAndPlotKernelBoundaryY(
    const SubLattice subLattice, real *__restrict__ df,
    real *__restrict__ df_tmp, real *__restrict__ dfT,
    real *__restrict__ dfT_tmp, real *__restrict__ plot,
    real *__restrict__ averageSrc, real *__restrict__ averageDst,
    const int *__restrict__ voxels, BoundaryCondition *__restrict__ bcs,
    const real nu, const real C, const real nuT, const real Pr_t,
    const real gBetta, const real Tref, const DisplayQuantity::Enum vis_q);

__global__ void ComputeAndPlotKernelBoundaryZ(
    const SubLattice subLattice, real *__restrict__ df,
    real *__restrict__ df_tmp, real *__restrict__ dfT,
    real *__restrict__ dfT_tmp, real *__restrict__ plot,
    real *__restrict__ averageSrc, real *__restrict__ averageDst,
    const int *__restrict__ voxels, BoundaryCondition *__restrict__ bcs,
    const real nu, const real C, const real nuT, const real Pr_t,
    const real gBetta, const real Tref, const DisplayQuantity::Enum vis_q);
