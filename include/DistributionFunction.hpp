#pragma once

#include "DistributionArray.hpp"

template <class T>
class DistributionFunction : public DistributionArray<T> {
 public:
  DistributionFunction(unsigned int Q, unsigned int latticeSizeX,
                       unsigned int latticeSizeY, unsigned int latticeSizeZ,
                       unsigned int subdivisions = 1);
};

#include "DistributionFunctionImpl.hpp"
