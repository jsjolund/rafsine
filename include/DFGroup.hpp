/**
  \file
      ArrayGPU.h
  \brief
      Helper class to define array on the GPU.
          Uses thrust vectors as a basis.
  \author
      Nicolas Delbosc, University of Leeds
*/

#pragma once

#include <thrust/copy.h>
#include <thrust/device_vector.h>
#include <thrust/fill.h>
#include <thrust/generate.h>

#include <algorithm>
#include <utility>

#include "CudaUtils.hpp"
#include "PartitionTopology.hpp"

// Define an group of array on the GPU
// designed for 3D arrays (but works in 2D as well)
// Useful to group all the distribution functions into a single array
// distribution functions (fi) are packed in memory based on their direction:
// memory: f1,f1,...,f1,f2,f2,...,f2,f3,f3,...
class DistributionFunctionsGroup : public Topology {
 private:
  // Number of arrays (or directions for distribution functions)
  const unsigned int m_Q;
  // Distribution functions on the GPU
  thrust::device_vector<real> m_dfGPU;
  // True if the arrays are also allocated and usable on the CPU
  bool m_useCPU;
  // Distribution functions on the CPU
  thrust::host_vector<real> m_dfCPU;

 public:
  // Constructor
  DistributionFunctionsGroup(unsigned int Q, unsigned int latticeSizeX,
                             unsigned int latticeSizeY,
                             unsigned int latticeSizeZ,
                             unsigned int subdivisions = 0, bool useCPU = true)
      : Topology(latticeSizeX, latticeSizeY, latticeSizeZ, subdivisions),
        m_Q(Q),
        m_useCPU(useCPU),
        m_dfCPU(useCPU ? Q * latticeSizeX * latticeSizeY * latticeSizeZ : 1),
        m_dfGPU(Q * latticeSizeX * latticeSizeY * latticeSizeZ) {}

  // Return the number of arrays in the group i.e. the number of distribution
  // functions
  inline unsigned int Q() { return m_Q; }

  // 1D access to distribution function on the CPU
  inline real &operator()(unsigned int df_idx, unsigned int idx) {
    if (m_useCPU) {
      return m_dfCPU[idx + df_idx * m_latticeSize.x * m_latticeSize.y *
                               m_latticeSize.z];
    } else {
      std::cerr << "Error in 'DistributionFunctionsGroup::operator()'  CPU "
                   "memory is not allocated."
                << std::endl;
      return m_dfCPU[0];
    }
  }

  // 3D access to distribution function on the CPU
  inline real &operator()(unsigned int df_idx, unsigned int x, unsigned int y,
                          unsigned int z = 0) {
    return (*this)(df_idx, x + y * m_latticeSize.x +
                               z * m_latticeSize.x * m_latticeSize.y);
  }

  // Upload the distributions functions from the CPU to the GPU
  inline DistributionFunctionsGroup &upload() {
    if (m_useCPU)
      m_dfGPU = m_dfCPU;
    else
      std::cerr << "Error in 'DistributionFunctionsGroup::upload()'  CPU "
                   "memory is not allocated."
                << std::endl;
    return *this;
  }

  // Download the distributions functions from the GPU to the CPU
  inline DistributionFunctionsGroup &download() {
    if (m_useCPU)
      m_dfCPU = m_dfGPU;
    else
      std::cerr << "Error in 'DistributionFunctionsGroup::upload()'  CPU "
                   "memory is not allocated."
                << std::endl;
    return *this;
  }

  // Return a pointer to the beginning of the GPU memory
  inline real *gpu_ptr(unsigned int df_idx = 0) {
    const int SIZE = m_latticeSize.x * m_latticeSize.y * m_latticeSize.z;
    return thrust::raw_pointer_cast(&(m_dfGPU)[df_idx * SIZE]);
  }

  // Copy from another group of distribution functions
  // SAME SIZE IS REQUIRED
  inline DistributionFunctionsGroup &operator=(
      const DistributionFunctionsGroup &f) {
    if ((m_Q == f.m_Q) && (m_latticeSize.x == f.m_latticeSize.x) &&
        (m_latticeSize.y == f.m_latticeSize.y) &&
        (m_latticeSize.z == f.m_latticeSize.z)) {
      if (m_useCPU)
        thrust::copy(f.m_dfCPU.begin(), f.m_dfCPU.end(), m_dfCPU.begin());
      thrust::copy(f.m_dfGPU.begin(), f.m_dfGPU.end(), m_dfGPU.begin());
    } else {
      std::cerr << "Error in 'DistributionFunctionsGroup::operator ='  sizes "
                   "do not match."
                << std::endl;
    }
    m_useCPU = f.m_useCPU;
    return *this;
  }

  // Static function to swap two DistributionFunctionsGroup
  static inline void swap(DistributionFunctionsGroup &f1,
                          DistributionFunctionsGroup &f2) {
    f1.m_dfCPU.swap(f2.m_dfCPU);
    f1.m_dfGPU.swap(f2.m_dfGPU);
  }

  // Fill the ith array, i.e. the ith distribution function with a constant
  // value for all nodes
  inline void fill(unsigned int df_idx, real value) {
    const int SIZE = m_latticeSize.x * m_latticeSize.y * m_latticeSize.z;
    if (m_useCPU)
      thrust::fill(m_dfCPU.begin() + df_idx * SIZE,
                   m_dfCPU.begin() + (df_idx + 1) * SIZE, value);
    thrust::fill(m_dfGPU.begin() + df_idx * SIZE,
                 m_dfGPU.begin() + (df_idx + 1) * SIZE, value);
  }
};
