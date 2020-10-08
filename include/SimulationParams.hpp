#pragma once

#include "CudaUtils.hpp"

/**
 * @brief Structure containing parameters for the CUDA kernel
 *
 */
class SimulationParams {
 public:
  real_t nu;      //!< Viscosity
  real_t C;       //!< Smagorinsky constant
  real_t nuT;     //!< Thermal diffusivity
  real_t Pr_t;    //!< Turbulent Prandtl number
  real_t gBetta;  //!< Gravity times thermal expansion
  real_t Tref;    //!< Reference temperature for Boussinesq
  real_t Tinit;   //!< Initial temperature

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
