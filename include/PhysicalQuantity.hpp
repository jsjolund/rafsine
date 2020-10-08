#pragma once

#include "CudaUtils.hpp"

typedef struct PhysicalQuantity {
  real_t rho;
  real_t T;
  real_t vx;
  real_t vy;
  real_t vz;
} PhysicalQuantity;

namespace LBM {
enum Enum { BGK, MRT };
}  // namespace LBM
