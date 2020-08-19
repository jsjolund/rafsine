#pragma once

#include "CudaUtils.hpp"

typedef struct PhysicalQuantity {
  real rho;
  real T;
  real vx;
  real vy;
  real vz;
} PhysicalQuantity;

namespace LBM {
enum Enum { BGK, MRT };
}  // namespace LBM
