#pragma once

#include "CudaUtils.hpp"

/**
 * @brief Structure containing parameters for the CUDA kernel
 *
 */
class SimulationParams {
 public:
  real nu;      //!< Viscosity
  real C;       //!< Smagorinsky constant
  real nuT;     //!< Thermal diffusivity
  real Pr_t;    //!< Turbulent Prandtl number
  real gBetta;  //!< Gravity times thermal expansion
  real Tref;    //!< Reference temperature for Boussinesq
  real Tinit;   //!< Initial temperature

  SimulationParams()
      : nu(0), C(0), nuT(0), Pr_t(0), gBetta(0), Tref(0), Tinit(0) {}

  explicit SimulationParams(const SimulationParams& kp)
      : nu(kp.nu),
        C(kp.C),
        nuT(kp.nuT),
        Pr_t(kp.Pr_t),
        gBetta(kp.gBetta),
        Tref(kp.Tref),
        Tinit(kp.Tinit) {}
};
