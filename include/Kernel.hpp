#pragma once

#include "BoundaryCondition.hpp"
#include "CFDScene.hpp"
#include "CudaMathHelper.h"
#include "CudaUtils.hpp"
#include "DdQq.hpp"
#include "PhysicalQuantity.hpp"

__device__ PhysicalQuantity compute(
    // Lattice position in partition
    const Eigen::Vector3i pos,
    // Size of partition
    const Eigen::Vector3i size,
    // Size of ghostLayer
    const Eigen::Vector3i ghostLayer,
    // Velocity distribution functions
    real* __restrict__ df,
    real* __restrict__ df_tmp,
    // Temperature distribution functions
    real* __restrict__ dfT,
    real* __restrict__ dfT_tmp,
    // Voxel type array
    const voxel_t* __restrict__ voxels,
    // Boundary condition data
    BoundaryCondition* __restrict__ bcs,
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
    real* __restrict__ averageDst);

__device__ void computeAndPlot(
    // Lattice position in partition
    const Eigen::Vector3i pos,
    // Size of partition
    const Eigen::Vector3i size,
    // Size of ghostLayer
    const Eigen::Vector3i ghostLayer,
    // Velocity distribution functions
    real* __restrict__ df,
    real* __restrict__ df_tmp,
    // Temperature distribution functions
    real* __restrict__ dfT,
    real* __restrict__ dfT_tmp,
    // Voxel type array
    const voxel_t* __restrict__ voxels,
    // Boundary condition data
    BoundaryCondition* __restrict__ bcs,
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
    // Quantity to be visualised
    const DisplayQuantity::Enum displayQuantity,
    // Plot array for display
    real* __restrict__ plot);

__global__ void ComputeKernelInterior(const Partition partition,
                                      real* __restrict__ df,
                                      real* __restrict__ df_tmp,
                                      real* __restrict__ dfT,
                                      real* __restrict__ dfT_tmp,
                                      const voxel_t* __restrict__ voxels,
                                      BoundaryCondition* __restrict__ bcs,
                                      const real nu,
                                      const real C,
                                      const real nuT,
                                      const real Pr_t,
                                      const real gBetta,
                                      const real Tref,
                                      real* __restrict__ averageSrc,
                                      real* __restrict__ averageDst);

__global__ void ComputeKernelBoundaryX(const Partition partition,
                                       real* __restrict__ df,
                                       real* __restrict__ df_tmp,
                                       real* __restrict__ dfT,
                                       real* __restrict__ dfT_tmp,
                                       const voxel_t* __restrict__ voxels,
                                       BoundaryCondition* __restrict__ bcs,
                                       const real nu,
                                       const real C,
                                       const real nuT,
                                       const real Pr_t,
                                       const real gBetta,
                                       const real Tref,
                                       real* __restrict__ averageSrc,
                                       real* __restrict__ averageDst);

__global__ void ComputeKernelBoundaryY(const Partition partition,
                                       real* __restrict__ df,
                                       real* __restrict__ df_tmp,
                                       real* __restrict__ dfT,
                                       real* __restrict__ dfT_tmp,
                                       const voxel_t* __restrict__ voxels,
                                       BoundaryCondition* __restrict__ bcs,
                                       const real nu,
                                       const real C,
                                       const real nuT,
                                       const real Pr_t,
                                       const real gBetta,
                                       const real Tref,
                                       real* __restrict__ averageSrc,
                                       real* __restrict__ averageDst);

__global__ void ComputeKernelBoundaryZ(const Partition partition,
                                       real* __restrict__ df,
                                       real* __restrict__ df_tmp,
                                       real* __restrict__ dfT,
                                       real* __restrict__ dfT_tmp,
                                       const voxel_t* __restrict__ voxels,
                                       BoundaryCondition* __restrict__ bcs,
                                       const real nu,
                                       const real C,
                                       const real nuT,
                                       const real Pr_t,
                                       const real gBetta,
                                       const real Tref,
                                       real* __restrict__ averageSrc,
                                       real* __restrict__ averageDst);

// Plotting

__global__ void ComputeAndPlotKernelInterior(
    const Partition partition,
    real* __restrict__ df,
    real* __restrict__ df_tmp,
    real* __restrict__ dfT,
    real* __restrict__ dfT_tmp,
    const voxel_t* __restrict__ voxels,
    BoundaryCondition* __restrict__ bcs,
    const real nu,
    const real C,
    const real nuT,
    const real Pr_t,
    const real gBetta,
    const real Tref,
    real* __restrict__ averageSrc,
    real* __restrict__ averageDst,
    const DisplayQuantity::Enum displayQuantity,
    real* __restrict__ plot);

__global__ void ComputeAndPlotKernelBoundaryX(
    const Partition partition,
    real* __restrict__ df,
    real* __restrict__ df_tmp,
    real* __restrict__ dfT,
    real* __restrict__ dfT_tmp,
    const voxel_t* __restrict__ voxels,
    BoundaryCondition* __restrict__ bcs,
    const real nu,
    const real C,
    const real nuT,
    const real Pr_t,
    const real gBetta,
    const real Tref,
    real* __restrict__ averageSrc,
    real* __restrict__ averageDst,
    const DisplayQuantity::Enum displayQuantity,
    real* __restrict__ plot);

__global__ void ComputeAndPlotKernelBoundaryY(
    const Partition partition,
    real* __restrict__ df,
    real* __restrict__ df_tmp,
    real* __restrict__ dfT,
    real* __restrict__ dfT_tmp,
    const voxel_t* __restrict__ voxels,
    BoundaryCondition* __restrict__ bcs,
    const real nu,
    const real C,
    const real nuT,
    const real Pr_t,
    const real gBetta,
    const real Tref,
    real* __restrict__ averageSrc,
    real* __restrict__ averageDst,
    const DisplayQuantity::Enum displayQuantity,
    real* __restrict__ plot);

__global__ void ComputeAndPlotKernelBoundaryZ(
    const Partition partition,
    real* __restrict__ df,
    real* __restrict__ df_tmp,
    real* __restrict__ dfT,
    real* __restrict__ dfT_tmp,
    const voxel_t* __restrict__ voxels,
    BoundaryCondition* __restrict__ bcs,
    const real nu,
    const real C,
    const real nuT,
    const real Pr_t,
    const real gBetta,
    const real Tref,
    real* __restrict__ averageSrc,
    real* __restrict__ averageDst,
    const DisplayQuantity::Enum displayQuantity,
    real* __restrict__ plot);
