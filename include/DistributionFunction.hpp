#pragma once

#include "DistributionArray.hpp"

class DistributionFunction : public DistributionArray<real> {
 public:
  DistributionFunction(unsigned int Q,
                       unsigned int latticeSizeX,
                       unsigned int latticeSizeY,
                       unsigned int latticeSizeZ,
                       unsigned int subdivisions,
                       D3Q4::Enum partitioning)
      : DistributionArray<real>(Q,
                                latticeSizeX,
                                latticeSizeY,
                                latticeSizeZ,
                                subdivisions,
                                1,
                                partitioning) {}
};
