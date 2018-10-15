#ifndef INCLUDE_KERNEL_HPP_
#define INCLUDE_KERNEL_HPP_

#include "CudaUtils.hpp"
#include "CudaMathHelper.h"
#include "CFDScene.hpp"
#include "BoundaryCondition.hpp"

const glm::ivec3 D3Q19directionVectors[26] = {
    // 6 faces
    glm::ivec3(1, 0, 0),
    glm::ivec3(-1, 0, 0),
    glm::ivec3(0, 1, 0),
    glm::ivec3(0, -1, 0),
    glm::ivec3(0, 0, 1),
    glm::ivec3(0, 0, -1),
    // 12 edges
    glm::ivec3(1, 1, 0),
    glm::ivec3(-1, -1, 0),
    glm::ivec3(1, -1, 0),
    glm::ivec3(-1, 1, 0),
    glm::ivec3(1, 0, 1),
    glm::ivec3(-1, 0, -1),
    glm::ivec3(1, 0, -1),
    glm::ivec3(-1, 0, 1),
    glm::ivec3(0, 1, 1),
    glm::ivec3(0, -1, -1),
    glm::ivec3(0, 1, -1),
    glm::ivec3(0, -1, 1),
    // 8 corners
    glm::ivec3(1, 1, 1),
    glm::ivec3(-1, -1, -1),
    glm::ivec3(-1, 1, 1),
    glm::ivec3(1, -1, -1),
    glm::ivec3(1, -1, 1),
    glm::ivec3(-1, 1, -1),
    glm::ivec3(1, 1, -1),
    glm::ivec3(-1, -1, 1),
};

__constant__ real D3Q19directions[19 * 3] = {
    0, 0, 0,
    // Main axis
    1, 0, 0,
    -1, 0, 0,
    0, 1, 0,
    0, -1, 0,
    0, 0, 1,
    0, 0, -1,
    // Diagonal
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
__constant__ int D3Q19directionsOpposite[19] = {
    0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15, 18, 17};
__constant__ real D3Q19weights[19] = {
    1.0f / 3.0f,
    1.0f / 18.0f,
    1.0f / 18.0f,
    1.0f / 18.0f,
    1.0f / 18.0f,
    1.0f / 18.0f,
    1.0f / 18.0f,
    1.0f / 36.0f,
    1.0f / 36.0f,
    1.0f / 36.0f,
    1.0f / 36.0f,
    1.0f / 36.0f,
    1.0f / 36.0f,
    1.0f / 36.0f,
    1.0f / 36.0f,
    1.0f / 36.0f,
    1.0f / 36.0f,
    1.0f / 36.0f,
    1.0f / 36.0f};

__constant__ real D3Q7directions[7 * 3] = {
    // Main axis
    0, 0, 0,
    1, 0, 0,
    -1, 0, 0,
    0, 1, 0,
    0, -1, 0,
    0, 0, 1,
    0, 0, -1};
__constant__ int D3Q7directionsOpposite[7] = {
    0, 2, 1, 4, 3, 6, 5};
__constant__ real D3Q7weights[7] = {
    0,
    1.0f / 6.0f,
    1.0f / 6.0f,
    1.0f / 6.0f,
    1.0f / 6.0f,
    1.0f / 6.0f,
    1.0f / 6.0f};

__global__ void
ComputeKernel(
    // Velocity distribution functions
    real *__restrict__ df,
    real *__restrict__ df_tmp,
    // Temperature distribution functions
    real *__restrict__ dfT,
    real *__restrict__ dfT_tmp,
    // Plot array for display
    real *__restrict__ plot,
    // Voxel type array
    const int *__restrict__ voxels,
    // Size of the domain
    const int nx, const int ny, const int nz,
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

#endif  // INCLUDE_KERNEL_HPP_"
