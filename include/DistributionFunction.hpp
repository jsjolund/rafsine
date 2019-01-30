#pragma once

#include "DistributionArray.hpp"

class DistributionFunction : public DistributionArray {
 public:
  DistributionFunction(unsigned int Q, unsigned int latticeSizeX,
                       unsigned int latticeSizeY, unsigned int latticeSizeZ,
                       unsigned int subdivisions = 1);
};
