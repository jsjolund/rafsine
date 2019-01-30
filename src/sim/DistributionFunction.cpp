#include "DistributionFunction.hpp"

DistributionFunction::DistributionFunction(unsigned int Q,
                                           unsigned int latticeSizeX,
                                           unsigned int latticeSizeY,
                                           unsigned int latticeSizeZ,
                                           unsigned int subdivisions)
    : DistributionArray(Q, latticeSizeX, latticeSizeY, latticeSizeZ,
                        subdivisions, 1) {}