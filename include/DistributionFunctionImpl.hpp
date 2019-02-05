#include "DistributionFunction.hpp"

template <class T>
DistributionFunction<T>::DistributionFunction(unsigned int Q,
                                              unsigned int latticeSizeX,
                                              unsigned int latticeSizeY,
                                              unsigned int latticeSizeZ,
                                              unsigned int subdivisions)
    : DistributionArray<T>(Q, latticeSizeX, latticeSizeY, latticeSizeZ,
                           subdivisions, 1) {}