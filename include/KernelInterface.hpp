#pragma once

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include <cuda_profiler_api.h>

#include <omp.h>

#include <vector>

#include "BoundaryCondition.hpp"
#include "DistributionFunction.hpp"
#include "Kernel.hpp"
#include "P2PLattice.hpp"

using thrust::device_vector;

/**
 * @brief Structure containing parameters for the CUDA kernel
 *
 */
class ComputeParams {
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

  DistributionFunction *avg; /*!< Contains the macroscopic temperature, velocity
                              * (x,y,z components) integrated in time (so
                              * /nbr_of_time_steps to get average) */

  VoxelArray *voxels;                     //!< The array of voxels
  device_vector<BoundaryCondition> *bcs;  //!< The boundary conditions

  ComputeParams()
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
        bcs(nullptr) {}

  ComputeParams &operator=(const ComputeParams &kp) {
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

  ~ComputeParams() {
    delete df;
    delete df_tmp;
    delete dfT;
    delete dfT_tmp;
    delete avg;
    delete voxels;
  }

  void allocate(SubLattice subLattice) {
    df->allocate(subLattice);
    df_tmp->allocate(subLattice);
    dfT->allocate(subLattice);
    dfT_tmp->allocate(subLattice);
    avg->allocate(subLattice);
  }
};

/**
 * @brief Class responsible for calling the CUDA kernel
 *
 */
class KernelInterface : public P2PLattice {
 private:
  // Cuda kernel parameters
  std::vector<ComputeParams *> m_params;

  void runComputeKernel(SubLattice subLattice, ComputeParams *kp,
                        real *plotGpuPointer,
                        DisplayQuantity::Enum displayQuantity,
                        cudaStream_t computeStream = 0);

 public:
  void runInitKernel(DistributionFunction *df, DistributionFunction *dfT,
                     SubLattice subLattice, float rho, float vx, float vy,
                     float vz, float T);
  void uploadBCs(BoundaryConditionsArray *bcs);
  void resetAverages();
  void resetDfs();
  void compute(real *plotGpuPtr, DisplayQuantity::Enum dispQ);
  KernelInterface(const ComputeParams *params,
                  const BoundaryConditionsArray *bcs, const VoxelArray *voxels,
                  const int numDevices);
  ~KernelInterface();
};
