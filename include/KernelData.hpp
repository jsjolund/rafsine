#pragma once

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include <cuda_profiler_api.h>

#include <omp.h>

#include <vector>

#include "BoundaryCondition.hpp"
#include "DistributedDFGroup.hpp"
#include "Kernel.hpp"

/**
 * @brief Structure containing parameters for the CUDA kernel
 *
 */
typedef struct KernelParameters {
  int nx;              //!< Size of the domain on X-axis
  int ny;              //!< Size of the domain on Y-axis
  int nz;              //!< Size of the domain on Z-axis
  real nu;             //!< Viscosity
  real C;              //!< Smagorinsky constant
  real nuT;            //!< Thermal diffusivity
  real Pr;             //!< Prandtl number of air
  real Pr_t;           //!< Turbulent Prandtl number
  real gBetta;         //!< Gravity times thermal expansion
  real Tref;           //!< Reference temperature for Boussinesq
  real Tinit;          //!< Initial temperature
  VoxelArray *voxels;  //!< The array of voxels
  thrust::device_vector<BoundaryCondition>
      *bcs;                     //!< Array of boundary conditions
  DistributedDFGroup *df;       //!< Velocity distribution functions
  DistributedDFGroup *df_tmp;   //!< Velocity distribution functions (for swap)
  DistributedDFGroup *dfT;      //!< Temp. distribution functions
  DistributedDFGroup *dfT_tmp;  //!< Temp. distribution functions (for swap)
  DistributedDFGroup *average;  //!< Contains the macroscopic temperature,
                                //!< velocity (x,y,z components) integrated in
                                //!< time (so /nbr_of_time_steps to get average)
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
  void initDomain(DistributedDFGroup *df, DistributedDFGroup *dfT,
                  Partition partition, float rho, float vx, float vy, float vz,
                  float T);
  void uploadBCs(BoundaryConditionsArray *bcs);
  void resetAverages();
  void compute(real *plotGpuPtr, DisplayQuantity::Enum dispQ);
  KernelData(const KernelParameters *params, const BoundaryConditionsArray *bcs,
             const VoxelArray *voxels, const int numDevices);
  ~KernelData();
};
