#pragma once

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include <cuda_profiler_api.h>

#include <omp.h>

#include <limits.h>
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
  DistributionFunction *dfT;     //!< Temperature distribution functions
  DistributionFunction *dfT_tmp;  //!< Temp. distribution functions (for swap)

  /**
   * Contains the macroscopic temperature, velocity (x,y,z components)
   * integrated in time, so divide by number of time steps to get average).
   * 0 -> temperature
   * 1 -> x-component of velocity
   * 2 -> y-component of velocity
   * 3 -> z-component of velocity
   */
  DistributionArray<real> *avg;
  /**
   * Plot array for slice renderer
   */
  DistributionArray<real> *plot;

  VoxelArray *voxels;                     //!< The array of voxels
  device_vector<BoundaryCondition> *bcs;  //!< The boundary conditions

  ComputeParams()
      : nu(0),
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
        avg(nullptr),
        voxels(nullptr),
        bcs(nullptr) {}

  explicit ComputeParams(const ComputeParams &kp)
      : nu(kp.nu),
        C(kp.C),
        nuT(kp.nuT),
        Pr(kp.Pr),
        Pr_t(kp.Pr_t),
        gBetta(kp.gBetta),
        Tref(kp.Tref),
        Tinit(kp.Tinit),
        df(kp.df),
        df_tmp(kp.df_tmp),
        dfT(kp.dfT),
        dfT_tmp(kp.dfT_tmp),
        avg(kp.avg),
        voxels(kp.voxels),
        bcs(kp.bcs) {}

  ~ComputeParams() {
    delete df;
    delete df_tmp;
    delete dfT;
    delete dfT_tmp;
    delete avg;
    delete voxels;
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
  DistributionArray<real> *m_avg;
  DistributionArray<real> *m_plot;
  int m_plotIndex;

  void runComputeKernel(SubLattice subLattice, ComputeParams *kp,
                        DisplayQuantity::Enum displayQuantity,
                        cudaStream_t computeStream = 0);
  void runInitKernel(DistributionFunction *df, DistributionFunction *dfT,
                     SubLattice subLattice, float rho, float vx, float vy,
                     float vz, float T);
  void exchange(int srcDev, SubLattice subLattice, D3Q7::Enum direction);

 public:
  void getMinMax(real *min, real *max);
  void uploadBCs(BoundaryConditionsArray *bcs);
  void resetAverages();
  void resetDfs();
  void compute(DisplayQuantity::Enum displayQuantity,
               glm::ivec3 slicePos = glm::ivec3(-1, -1, -1));
  void plot(int plotDev, thrust::device_vector<real> *plot);

  real *gpu_ptr() {
    return m_plot->gpu_ptr(m_plot->getSubLattice(0, 0, 0), m_plotIndex);
  }

  KernelInterface(const int nx, const int ny, const int nz,
                  const ComputeParams *params,
                  const BoundaryConditionsArray *bcs, const VoxelArray *voxels,
                  const int numDevices);
  ~KernelInterface();
};
