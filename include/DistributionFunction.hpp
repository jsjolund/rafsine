#pragma once

#include "DistributionArray.hpp"

class DistributionFunction : public DistributionArray<real_t> {
 public:
  DistributionFunction(unsigned int Q,
                       unsigned int nx,
                       unsigned int ny,
                       unsigned int nz,
                       unsigned int nd,
                       D3Q4::Enum partitioning)
      : DistributionArray<real_t>(Q, nx, ny, nz, nd, 1, partitioning) {}
};
