#pragma once

#include "DistributionArray.hpp"

class DistributionFunction : public DistributionArray<real> {
 public:
  DistributionFunction(unsigned int Q, unsigned int latticeSizeX,
                       unsigned int latticeSizeY, unsigned int latticeSizeZ,
                       unsigned int subdivisions = 1)
      : DistributionArray<real>(Q, latticeSizeX, latticeSizeY, latticeSizeZ,
                                subdivisions, 1) {}
};
