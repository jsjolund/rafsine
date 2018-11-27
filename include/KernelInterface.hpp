#pragma once

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include <cuda_profiler_api.h>

#include <omp.h>

#include <vector>

#include "BoundaryCondition.hpp"
#include "DistributionFunction.hpp"
#include "Kernel.hpp"

/**
 * @brief Structure containing parameters for the CUDA kernel
 *
 */
class KernelParameters {
 public:
  int nx;  //!< Size of the domain on X-axis
  int ny;  //!< Size of the domain on Y-axis
  int nz;  //!< Size of the domain on Z-axis

  real nu;      //!< Viscosity
  real C;       //!< Smagorinsky constant
  real nuT;     //!< Thermal diffusivity
  real Pr;      //!< Prandtl number of air
  real Pr_t;    //!< Turbulent Prandtl number
  real gBetta;  //!< Gravity times thermal expansion
  real Tref;    //!< Reference temperature for Boussinesq
  real Tinit;   //!< Initial temperature

  DistributionFunction *df;      //!< Velocity distribution functions
  DistributionFunction *df_tmp;  //!< Velocity distribution functions (for swap)
  DistributionFunction *dfT;     //!< Temp. distribution functions
  DistributionFunction *dfT_tmp;  //!< Temp. distribution functions (for swap)

  /*!< Contains the macroscopic temperature, velocity (x,y,z components)
   * integrated in time (so /nbr_of_time_steps to get average) */
  DistributionFunction *avg;

  VoxelArray *voxels;                             //!< The array of voxels
  thrust::device_vector<BoundaryCondition> *bcs;  //!< The boundary conditions

  std::vector<bool> *peerAccessList;
  cudaStream_t streams[27];

  KernelParameters()
      : nx(0),
        ny(0),
        nz(0),
        nu(0),
        C(0),
        nuT(0),
        Pr(0),
        Pr_t(0),
        gBetta(0),
        Tref(0),
        Tinit(0),
        df(nullptr),
        df_tmp(nullptr),
        dfT(nullptr),
        dfT_tmp(nullptr),
        voxels(nullptr),
        bcs(nullptr),
        peerAccessList(nullptr) {}

  KernelParameters &operator=(const KernelParameters &kp) {
    nx = kp.nx;
    ny = kp.ny;
    nz = kp.nz;
    nu = kp.nu;
    C = kp.C;
    nuT = kp.nuT;
    Pr = kp.Pr;
    Pr_t = kp.Pr_t;
    gBetta = kp.gBetta;
    Tref = kp.Tref;
    Tinit = kp.Tinit;
    return *this;
  }

  ~KernelParameters() {
    delete peerAccessList;
    delete df;
    delete df_tmp;
    delete dfT;
    delete dfT_tmp;
    delete avg;
    delete voxels;
  }

  void init(const VoxelArray *voxelArray,
            const BoundaryConditionsArray *boundaryConditions) {
    for (int q = 0; q < 4; q++) avg->fill(q, 0);
    avg->upload();

    voxels = new VoxelArray(*voxelArray);
    voxels->upload();
    bcs = new thrust::device_vector<BoundaryCondition>(*boundaryConditions);
  }

  void allocate(Partition partition) {
    df->allocate(partition);
    df_tmp->allocate(partition);
    dfT->allocate(partition);
    dfT_tmp->allocate(partition);
    avg->allocate(partition);
  }
};

/**
 * @brief Class responsible for calling the CUDA kernel
 *
 */
class KernelInterface {
 private:
  // Number of CUDA devices
  int m_numDevices;

  // Cuda kernel parameters
  std::vector<KernelParameters *> m_params;

  std::unordered_map<Partition, int> m_partitionDeviceMap;
  std::vector<std::vector<Partition>> m_devicePartitionMap;

  void runComputeKernel(Partition partition, KernelParameters *kp,
                        real *plotGpuPointer,
                        DisplayQuantity::Enum displayQuantity,
                        cudaStream_t computeStream);

  void runHaloExchangeKernel(Partition partition, KernelParameters *kp);
  bool enablePeerAccess(int srcDev, int dstDev,
                        std::vector<bool> *peerAccessList);
  void disablePeerAccess(int srcDev, std::vector<bool> *peerAccessList);

 public:
  void runInitKernel(DistributionFunction *df, DistributionFunction *dfT,
                     Partition partition, float rho, float vx, float vy,
                     float vz, float T);
  void uploadBCs(BoundaryConditionsArray *bcs);
  void resetAverages();
  void compute(real *plotGpuPtr, DisplayQuantity::Enum dispQ);
  KernelInterface(const KernelParameters *params,
                  const BoundaryConditionsArray *bcs, const VoxelArray *voxels,
                  const int numDevices);
  ~KernelInterface();
};
