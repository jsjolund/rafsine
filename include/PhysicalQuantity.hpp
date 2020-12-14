#pragma once

#include "CudaUtils.hpp"

/**
 * @brief Macroscopic values calculated by LBM kernel used for plotting and
 * averaging
 */
typedef struct PhysicalQuantity {
  //! Density
  real_t rho;
  //! Temperature
  real_t T;
  //! Velocity along X-axis
  real_t vx;
  //! Velocity along Y-axis
  real_t vy;
  //! Velocity along Z-axis
  real_t vz;
} PhysicalQuantity;
