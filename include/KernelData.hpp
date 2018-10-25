#pragma once

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include <omp.h>

#include <vector>

#include "BoundaryCondition.hpp"
#include "DFGroup.hpp"
#include "DistributedDFGroup.hpp"
#include "Kernel.hpp"

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
  // CUDA streams
  std::vector<cudaStream_t> streams;
  // Velocity distribution functions
  DistributedDFGroup *df, *df_tmp;
  // Temperature distribution functions
  DistributedDFGroup *dfT, *dfT_tmp;
  // Contains the macroscopic temperature, velocity (x,y,z components)
  // integrated in time (so /nbr_of_time_steps to get average)
  DistributedDFGroup *average;
  // The array of voxels
  VoxelArray *voxels;
  // Array of boundary conditions
  thrust::device_vector<BoundaryCondition> *bcs;
} KernelParameters;

/**
 * @brief Class responsible for calling the CUDA kernel
 *
 */
class KernelData {
 private:
  // Number of CUDA devices
  int m_numDevices;

  // Cuda kernel parameters
  std::vector<KernelParameters> m_params;

  std::unordered_map<Partition, int> m_partitionDeviceMap;
  std::vector<std::vector<Partition>> m_devicePartitionMap;

 public:
  void initDomain(DistributedDFGroup *df, DistributedDFGroup *dfT, float rho,
                  float vx, float vy, float vz, float T);
  void uploadBCs(BoundaryConditionsArray *bcs);
  void resetAverages();
  void compute(real *plotGpuPtr, DisplayQuantity::Enum dispQ);
  KernelData(const KernelParameters *params, const BoundaryConditionsArray *bcs,
             const VoxelArray *voxels, const int numDevices);
  ~KernelData();
};
