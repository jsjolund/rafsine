#pragma once

#include "CudaUtils.hpp"

/**
 * @brief Structure containing parameters for the CUDA kernel
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

  /**
   * @brief Copy constructor
   *
   * @param params
   */
  explicit SimulationParams(const SimulationParams& params)
      : nu(params.nu),
        C(params.C),
        nuT(params.nuT),
        Pr_t(params.Pr_t),
        gBetta(params.gBetta),
        Tref(params.Tref),
        Tinit(params.Tinit) {}
};
