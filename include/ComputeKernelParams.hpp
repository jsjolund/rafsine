#pragma once

#include "BoundaryCondition.hpp"
#include "DistributionFunction.hpp"
#include "VoxelArray.hpp"

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
  //! Value map for average gathering
  thrust::device_vector<int> *avg_map;
  //! Stencil for average gathering
  thrust::device_vector<int> *avg_stencil;
  //! Plot array for slice renderer
  DistributionArray<real> *plot;
  //! The array of voxels
  VoxelArray *voxels;
  //! The boundary conditions
  thrust::device_vector<BoundaryCondition> *bcs;

  ~ComputeParams() {
    delete df;
    delete df_tmp;
    delete dfT;
    delete dfT_tmp;
    delete avg;
    delete avg_map;
    delete avg_stencil;
    delete plot;
    delete voxels;
    delete bcs;
  }

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
        avg_map(nullptr),
        avg_stencil(nullptr),
        plot(nullptr),
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
        avg_map(kp.avg_map),
        avg_stencil(kp.avg_stencil),
        plot(kp.plot),
        voxels(kp.voxels),
        bcs(kp.bcs) {}
};
