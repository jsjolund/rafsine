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
class DistributedDFGroup : public Topology {
 private:
  // Number of arrays (or directions for distribution functions)
  const unsigned int m_Q;
  // Distribution functions on the GPU
  std::unordered_map<Partition, thrust::device_vector<real>*> m_dfGPU;
  // Distribution functions on the CPU
  std::unordered_map<Partition, thrust::host_vector<real>*> m_dfCPU;

 public:
  // Constructor
  DistributedDFGroup(unsigned int Q, unsigned int latticeSizeX,
                     unsigned int latticeSizeY, unsigned int latticeSizeZ,
                     unsigned int subdivisions = 0)
      : Topology(latticeSizeX, latticeSizeY, latticeSizeZ, subdivisions),
        m_Q(Q) {}

  // Return the number of arrays in the group i.e. the number of distribution
  // functions
  inline unsigned int Q() { return m_Q; }

  inline void allocate(Partition p) {
    glm::ivec3 pSize = p.getLatticeSize();
    int size = (pSize.x + 2) * (pSize.y + 2) * (pSize.z + 2) * m_Q;
    thrust::device_vector<real>* dfGPU = new thrust::device_vector<real>(size);
    thrust::host_vector<real>* dfCPU = new thrust::host_vector<real>(size);
    m_dfGPU[p] = dfGPU;
    m_dfCPU[p] = dfCPU;
  }

  inline unsigned int64 memoryUse() {
    int sum = 0;
    for (std::pair<Partition, thrust::device_vector<real>*> element : m_dfGPU)
      sum += element.second->size() * sizeof(real);
    return sum;
  }

  // Fill the ith array, i.e. the ith distribution function with a constant
  // value for all nodes
  inline void fill(unsigned int df_idx, real value) {
    for (std::pair<Partition, thrust::device_vector<real>*> element : m_dfGPU) {
      const glm::ivec3 pSize = element.first.getLatticeSize();
      const int size = (pSize.x + 2) * (pSize.y + 2) * (pSize.z + 2);
      thrust::device_vector<real>* dfGPU = element.second;
      thrust::fill(dfGPU->begin() + df_idx * size,
                   dfGPU->begin() + (df_idx + 1) * size, value);
    }
    for (std::pair<Partition, thrust::host_vector<real>*> element : m_dfCPU) {
      const glm::ivec3 pSize = element.first.getLatticeSize();
      const int size = (pSize.x + 2) * (pSize.y + 2) * (pSize.z + 2);
      thrust::host_vector<real>* dfGPU = element.second;
      thrust::fill(dfGPU->begin() + df_idx * size,
                   dfGPU->begin() + (df_idx + 1) * size, value);
    }
  }

  // // 1D access to distribution function on the CPU
  // inline real &operator()(unsigned int df_idx, unsigned int idx) {
  //   return m_dfCPU[idx + df_idx * m_latticeSize.x * m_latticeSize.y *
  //                            m_latticeSize.z];
  // }

  // // 3D access to distribution function on the CPU
  // inline real &operator()(unsigned int df_idx, unsigned int x, unsigned int
  // y,
  //                         unsigned int z = 0) {
  //   return (*this)(df_idx, x + y * m_latticeSize.x +
  //                              z * m_latticeSize.x * m_latticeSize.y);
  // }

  // // Upload the distributions functions from the CPU to the GPU
  // inline DistributedDFGroup &upload() {
  //   m_dfGPU = m_dfCPU;

  //   return *this;
  // }

  // // Download the distributions functions from the GPU to the CPU
  // inline DistributedDFGroup &download() {
  //   m_dfCPU = m_dfGPU;
  //   return *this;
  // }

  // // Return a pointer to the beginning of the GPU memory
  // inline real *gpu_ptr(unsigned int df_idx = 0) {
  //   const int SIZE = m_latticeSize.x * m_latticeSize.y * m_latticeSize.z;
  //   return thrust::raw_pointer_cast(&(m_dfGPU)[df_idx * SIZE]);
  // }

  // // Copy from another group of distribution functions
  // // SAME SIZE IS REQUIRED
  // inline DistributedDFGroup &operator=(const DistributedDFGroup &f) {
  //   if ((m_Q == f.m_Q) && (m_latticeSize.x == f.m_latticeSize.x) &&
  //       (m_latticeSize.y == f.m_latticeSize.y) &&
  //       (m_latticeSize.z == f.m_latticeSize.z)) {
  //     thrust::copy(f.m_dfCPU.begin(), f.m_dfCPU.end(), m_dfCPU.begin());
  //     thrust::copy(f.m_dfGPU.begin(), f.m_dfGPU.end(), m_dfGPU.begin());
  //   } else {
  //     std::cerr << "Error in 'DistributionFunctionsGroup::operator ='  sizes
  //     "
  //                  "do not match."
  //               << std::endl;
  //   }
  //   return *this;
  // }

  // // Static function to swap two DistributionFunctionsGroup
  // static inline void swap(DistributedDFGroup &f1, DistributedDFGroup &f2) {
  //   f1.m_dfCPU.swap(f2.m_dfCPU);
  //   f1.m_dfGPU.swap(f2.m_dfGPU);
  // }
};
