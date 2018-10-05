#pragma once

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include <omp.h>

#include "BoundaryCondition.hpp"
#include "DistributionFunctionsGroup.hpp"
#include "Kernel.hpp"

extern cudaStream_t simStream;

typedef struct KernelParameters {
  // Size of the domain
  int nx, ny, nz;
  // Viscosity
  real nu;
  // Smagorinsky constant
  real C;
  // Thermal diffusivity
  real nuT;
  // Prandtl number of air
  real Pr;
  // Turbulent Prandtl number
  real Pr_t;
  // Gravity times thermal expansion
  real gBetta;
  // Reference temperature for Boussinesq
  real Tref;
  real Tinit;
} KernelParameters;

class KernelData {
 private:
  thrust::device_vector<BoundaryCondition> *m_bcs_d;
  inline BoundaryCondition *bcs_gpu_ptr() {
    return thrust::raw_pointer_cast(&(*m_bcs_d)[0]);
  }

 public:
  // Cuda kernel parameters
  dim3 *m_grid_size, *m_block_size;
  // Cuda stream for simulation
  cudaStream_t m_simStream;

  KernelParameters *m_params;
  VoxelArray *m_voxels;

  // Velocity distribution functions
  DistributionFunctionsGroup *m_df, *m_df_tmp;

  // Temperature distribution functions
  DistributionFunctionsGroup *m_dfT, *m_dfT_tmp;

  // Contains the macroscopic temperature, velocity (x,y,z components)
  // integrated in time (so /nbr_of_time_steps to get average)
  DistributionFunctionsGroup *m_average;

  void initDomain(float rho, float vx, float vy, float vz, float T);
  void uploadBCs(BoundaryConditionsArray *bcs);
  void resetAverages();
  void compute(real *plotGpuPtr, DisplayQuantity::Enum dispQ,
               cudaStream_t simStream);
  KernelData(KernelParameters *params, BoundaryConditionsArray *bcs,
             VoxelArray *voxels);
  ~KernelData();
};
