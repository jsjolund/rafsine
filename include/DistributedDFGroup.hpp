#pragma once

#include <thrust/copy.h>
#include <thrust/device_vector.h>
#include <thrust/fill.h>
#include <thrust/generate.h>

#include <algorithm>
#include <iostream>
#include <string>
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
  struct thrust_vectors {
    thrust::device_vector<real>* gpu;
    thrust::host_vector<real>* cpu;
  };
  // Distribution functions on
  std::unordered_map<Partition, thrust_vectors> m_df;

 public:
  // Constructor
  DistributedDFGroup(unsigned int Q, unsigned int latticeSizeX,
                     unsigned int latticeSizeY, unsigned int latticeSizeZ,
                     unsigned int subdivisions = 0)
      : Topology(latticeSizeX, latticeSizeY, latticeSizeZ, subdivisions),
        m_Q(Q) {}

  // Return the number of arrays in the group i.e. the number of distribution
  // functions
  inline unsigned int getQ() const { return m_Q; }

  inline void allocate(Partition p) {
    glm::ivec3 pSize = p.getLatticeSize();
    int size = (pSize.x + 2) * (pSize.y + 2) * (pSize.z + 2) * m_Q;
    m_df[p] = {.gpu = new thrust::device_vector<real>(size),
               .cpu = new thrust::host_vector<real>(size)};
  }

  inline unsigned long memoryUse() {
    int sum = 0;
    for (std::pair<Partition, thrust_vectors> element : m_df)
      sum += element.second.cpu->size() * sizeof(real);
    return sum;
  }

  // Fill the ith array, i.e. the ith distribution function with a constant
  // value for all nodes
  inline void fill(unsigned int df_idx, real value) {
    for (std::pair<Partition, thrust_vectors> element : m_df) {
      const glm::ivec3 pSize = element.first.getLatticeSize();
      const int size = (pSize.x + 2) * (pSize.y + 2) * (pSize.z + 2);
      thrust::device_vector<real>* dfGPU = element.second.gpu;
      thrust::fill(dfGPU->begin() + df_idx * size,
                   dfGPU->begin() + (df_idx + 1) * size, value);
      thrust::host_vector<real>* dfCPU = element.second.cpu;
      thrust::fill(dfCPU->begin() + df_idx * size,
                   dfCPU->begin() + (df_idx + 1) * size, value);
    }
  }

  // 3D access to DF on the CPU (global coordinates without halos)
  inline real& global(int df_idx, int x, int y, int z = 0) {
    glm::ivec3 p(x, y, z);
    glm::ivec3 min, max, n;
    for (std::pair<Partition, thrust_vectors> element : m_df) {
      min = element.first.getLatticeMin();
      max = element.first.getLatticeMax();
      n = max - min + glm::ivec3(2, 2, 2);
      if (p.x >= min.x && p.y >= min.y && p.z >= min.z && p.x < max.x &&
          p.y < max.y && p.z < max.z) {
        p = p - min + glm::ivec3(1, 1, 1);
        int idx = p.x + p.y * n.x + p.z * n.x * n.y + df_idx * n.x * n.y * n.z;
        assert(element.second.cpu->size() == n.x * n.y * n.z);
        assert(idx < element.second.cpu->size());
        return (*element.second.cpu)[idx];
      }
    }
    throw std::out_of_range("Invalid range global");
  }

  inline real globalConst(int df_idx, int x, int y, int z = 0) const {
    glm::ivec3 p(x, y, z);
    glm::ivec3 min, max, n;
    for (std::pair<Partition, thrust_vectors> element : m_df) {
      min = element.first.getLatticeMin();
      max = element.first.getLatticeMax();
      n = max - min + glm::ivec3(2, 2, 2);
      if (p.x >= min.x && p.y >= min.y && p.z >= min.z && p.x < max.x &&
          p.y < max.y && p.z < max.z) {
        p = p - min + glm::ivec3(1, 1, 1);
        int idx = p.x + p.y * n.x + p.z * n.x * n.y + df_idx * n.x * n.y * n.z;
        assert(element.second.cpu->size() == n.x * n.y * n.z);
        assert(idx < element.second.cpu->size());
        return (*element.second.cpu)[idx];
      }
    }
    throw std::out_of_range("Invalid range global");
  }

  // 3D access to DF on the CPU (local coordinates with halos)
  // start at -1 end at n + 1
  inline real& local(int df_idx, int x, int y, int z = 0) {
    glm::ivec3 p(x, y, z);
    glm::ivec3 min, max, n;
    for (std::pair<Partition, thrust_vectors> element : m_df) {
      min = element.first.getLatticeMin() - glm::ivec3(1, 1, 1);
      max = element.first.getLatticeMax() + glm::ivec3(1, 1, 1);
      n = element.first.getLatticeSize() + glm::ivec3(2, 2, 2);
      if (p.x >= min.x && p.y >= min.y && p.z >= min.z && p.x < max.x &&
          p.y < max.y && p.z < max.z) {
        p = p - element.first.getLatticeMin() + glm::ivec3(1, 1, 1);
        int idx = p.x + p.y * n.x + p.z * n.x * n.y + df_idx * n.x * n.y * n.z;
        assert(element.second.cpu->size() == n.x * n.y * n.z);
        assert(idx < element.second.cpu->size());
        return (*element.second.cpu)[idx];
      }
    }
    throw std::out_of_range("Invalid range local");
  }

  inline real localConst(int df_idx, int x, int y, int z = 0) const {
    glm::ivec3 p(x, y, z);
    glm::ivec3 min, max, n;
    for (std::pair<Partition, thrust_vectors> element : m_df) {
      min = element.first.getLatticeMin() - glm::ivec3(1, 1, 1);
      max = element.first.getLatticeMax() + glm::ivec3(1, 1, 1);
      n = element.first.getLatticeSize() + glm::ivec3(2, 2, 2);
      if (p.x >= min.x && p.y >= min.y && p.z >= min.z && p.x < max.x &&
          p.y < max.y && p.z < max.z) {
        p = p - element.first.getLatticeMin() + glm::ivec3(1, 1, 1);
        int idx = p.x + p.y * n.x + p.z * n.x * n.y + df_idx * n.x * n.y * n.z;
        assert(element.second.cpu->size() == n.x * n.y * n.z);
        assert(idx < element.second.cpu->size());
        return (*element.second.cpu)[idx];
      }
    }
    throw std::out_of_range("Invalid range local readonly");
  }

  // Upload the distributions functions from the CPU to the GPU
  inline DistributedDFGroup& upload() {
    for (std::pair<Partition, thrust_vectors> element : m_df) {
      *element.second.gpu = *element.second.cpu;
    }
    return *this;
  }

  // Download the distributions functions from the GPU to the CPU
  inline DistributedDFGroup& download() {
    for (std::pair<Partition, thrust_vectors> element : m_df) {
      *element.second.cpu = *element.second.gpu;
    }
    return *this;
  }

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

std::ostream& operator<<(std::ostream& os, const DistributedDFGroup& df) {
  glm::ivec3 min = glm::ivec3(-1, -1, -1);
  glm::ivec3 max = df.getLatticeSize() + glm::ivec3(1, 1, 1);
  for (int q = 0; q < df.getQ(); q++) {
    for (int z = min.z; z < max.z; z++) {
      for (int y = min.y; y < max.y; y++) {
        for (int x = min.x; x < max.x; x++) {
          try {
            real value = df.localConst(q, x, y, z);
            os << value;
          } catch (std::out_of_range& e) {
            os << "X";
          }
          if (x < max.x - 1) os << ",";
        }
        os << std::endl;
      }
      os << std::endl;
    }
  }
  return os;
}
