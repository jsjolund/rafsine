#pragma once

#include "CudaUtils.hpp"
#include "CudaMathHelper.h"
#include "BoundaryCondition.hpp"
#include "CFDScene.hpp"

__constant__ real D3Q19directions[19 * 3] = {
    0, 0, 0,
    //main axis
    1, 0, 0,
    -1, 0, 0,
    0, 1, 0,
    0, -1, 0,
    0, 0, 1,
    0, 0, -1,
    //diagonal
    1, 1, 0,
    -1, -1, 0,
    1, -1, 0,
    -1, 1, 0,
    1, 0, 1,
    -1, 0, -1,
    1, 0, -1,
    -1, 0, 1,
    0, 1, 1,
    0, -1, -1,
    0, 1, -1,
    0, -1, 1};
__constant__ real D3Q6directions[6 * 3] = {
    //main axis
    1, 0, 0,
    -1, 0, 0,
    0, 1, 0,
    0, -1, 0,
    0, 0, 1,
    0, 0, -1};
__constant__ int D3Q19directionsOpposite[19] = {
    0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15, 18, 17};
__constant__ real D3Q19weights[19] = {
    1.0f / 3,
    1.0f / 18,
    1.0f / 18,
    1.0f / 18,
    1.0f / 18,
    1.0f / 18,
    1.0f / 18,
    1.0f / 36,
    1.0f / 36,
    1.0f / 36,
    1.0f / 36,
    1.0f / 36,
    1.0f / 36,
    1.0f / 36,
    1.0f / 36,
    1.0f / 36,
    1.0f / 36,
    1.0f / 36,
    1.0f / 36};
__constant__ real D3Q6weight[1] = {1.0f / 6};

void
    __global__
    ComputeKernel(
        //velocituy distribution functions
        real *df, real *df_tmp,
        //temperature distribution functions
        real *dfT, real *dfT_tmp,
        //plot array for display
        real *plot,
        //voxel type array
        int *voxels,
        //size of the domain
        int nx, int ny, int nz,
        //viscosity
        real nu,
        //Smagorinsky constant
        real C,
        //thermal diffusivity
        real nuT,
        //Turbulent Prandtl number
        real Pr_t,
        //gravity times thermal expansion
        real gBetta,
        //reference temperature for Boussinesq
        real Tref,
        //quantity to be visualised
        DisplayQuantity::Enum vis_q,
        //contain the macroscopic temperature, velocity (x,y,z components)
        //  integrated in time (so /nbr_of_time_steps to get average)
        real *average,
        // Boundary condition data
        BoundaryCondition *bcs);
