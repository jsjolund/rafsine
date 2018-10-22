#pragma once

#include <thrust/copy.h>
#include <thrust/device_vector.h>
#include <thrust/fill.h>
#include <thrust/generate.h>

#include <algorithm>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

#include "CudaUtils.hpp"
#include "PartitionTopology.hpp"

// Define a group of array on the GPU
// designed for 3D arrays (but works in 2D as well)
// Useful to group all the distribution functions into a single array
// distribution functions (fi) are packed in memory based on their direction:
// memory: f1,f1,...,f1,f2,f2,...,f2,f3,f3,...
class DistributedDFGroup : public Topology {
 private:
  struct thrust_vectors {
    thrust::device_vector<real>* gpu;
    thrust::host_vector<real>* cpu;
  };

  // Number of arrays (or directions for distribution functions)
  const unsigned int m_Q;

  // Distribution functions on
  std::unordered_map<Partition, thrust_vectors> m_df;

 public:
  // Constructor
  DistributedDFGroup(unsigned int Q, unsigned int latticeSizeX,
                     unsigned int latticeSizeY, unsigned int latticeSizeZ,
                     unsigned int subdivisions = 0)
      : Topology(latticeSizeX, latticeSizeY, latticeSizeZ, subdivisions),
        m_Q(Q) {}

  ~DistributedDFGroup() {
    for (std::pair<Partition, thrust_vectors> element : m_df) {
      delete element.second.gpu;
      delete element.second.cpu;
    }
  }

  // Return the number of arrays in the group i.e. the number of distribution
  // functions
  inline unsigned int getQ() const { return m_Q; }

  inline void allocate(Partition p) {
    glm::ivec3 pSize = p.getLatticeSize();
    int size = (pSize.x + 2) * (pSize.y + 2) * (pSize.z + 2) * m_Q;
    m_df[p] = {.gpu = new thrust::device_vector<real>(size),
               .cpu = new thrust::host_vector<real>(size)};
  }

  inline std::vector<Partition> getAllocatedPartitions() {
    std::vector<Partition> partitions;
    for (std::pair<Partition, thrust_vectors> element : m_df) {
      partitions.push_back(element.first);
    }
    return partitions;
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

  // Read/write to allocated partitions, excluding halos
  inline real& operator()(unsigned int df_idx, unsigned int x, unsigned int y,
                          unsigned int z = 0) {
    glm::ivec3 p(x, y, z);
    for (std::pair<Partition, thrust_vectors> element : m_df) {
      Partition partition = element.first;
      thrust_vectors vec = element.second;
      glm::ivec3 min = partition.getLatticeMin();
      glm::ivec3 max = partition.getLatticeMax();
      glm::ivec3 n = partition.getLatticeSize() + glm::ivec3(2, 2, 2);
      if (p.x >= min.x && p.y >= min.y && p.z >= min.z && p.x < max.x &&
          p.y < max.y && p.z < max.z && df_idx < m_Q) {
        glm::ivec3 q = p - partition.getLatticeMin() + glm::ivec3(1, 1, 1);
        int idx = I4D(df_idx, q.x, q.y, q.z, n.x, n.y, n.z);
        assert(vec.cpu->size() == n.x * n.y * n.z * m_Q);
        assert(idx < vec.cpu->size());
        return (*vec.cpu)[idx];
      }
    }
    throw std::out_of_range("Invalid range");
  }

  // Read/write to specific allocated partition, including halos
  // start at -1 end at n + 1
  inline real& operator()(Partition partition, unsigned int df_idx, int x,
                          int y, int z = 0) {
    if (m_df.find(partition) == m_df.end())
      throw std::out_of_range("Partition not allocated");
    thrust::host_vector<real>* cpuVector = m_df[partition].cpu;
    int idx = partition.toLocalIndex(df_idx, x, y, z);
    if (idx == -1)
      throw std::out_of_range("Invalid range");
    else
      return (*cpuVector)[idx];
  }

  // Return a pointer to the beginning of the GPU memory
  inline real* gpu_ptr(Partition partition, unsigned int idx = 0) {
    if (m_df.find(partition) == m_df.end())
      throw std::out_of_range("Partition not allocated");
    thrust::device_vector<real>* gpuVector = m_df[partition].gpu;
    return thrust::raw_pointer_cast(&(*gpuVector)[idx]);
  }

  // Return a pointer to the beginning of the GPU memory
  inline real* gpu_ptr(Partition partition, unsigned int df_idx, int x, int y,
                       int z) {
    if (m_df.find(partition) == m_df.end())
      throw std::out_of_range("Partition not allocated");
    thrust::device_vector<real>* gpuVector = m_df[partition].gpu;
    int idx = partition.toLocalIndex(df_idx, x, y, z);
    if (idx == -1)
      throw std::out_of_range("Invalid range");
    else
      return thrust::raw_pointer_cast(&(*gpuVector)[idx]);
  }

  inline void haloExchange(int devId, Partition partition,  int nDevId,
                           DistributedDFGroup* nDf, Partition neighbour,
                           std::vector<int>* srcIndex,
                           std::vector<int>* dstIndex, cudaStream_t cpyStream) {
    for (int j = 0; j < srcIndex->size(); j++) {
      // TODO(Only take the relevant direction vector)
      for (int q = 0; q < m_Q; ++q) {
        int srcOffset = q * partition.getDFSize().x * partition.getDFSize().y *
                        partition.getDFSize().z;
        int dstOffset = q * neighbour.getDFSize().x * neighbour.getDFSize().y *
                        neighbour.getDFSize().z;
        real* srcPtr = gpu_ptr(partition, srcIndex->at(j) + srcOffset);
        real* dstPtr = nDf->gpu_ptr(neighbour, dstIndex->at(j) + dstOffset);
        size_t size = sizeof(real);
        if (nDevId == devId) {
          CUDA_RT_CALL(cudaMemcpyAsync(dstPtr, srcPtr, size,
                                       cudaMemcpyDeviceToDevice, cpyStream));
        } else {
          CUDA_RT_CALL(cudaMemcpyPeerAsync(dstPtr, nDevId, srcPtr, devId, size,
                                           cpyStream));
        }
      }
    }
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

  inline unsigned long memoryUse() {
    int sum = 0;
    for (std::pair<Partition, thrust_vectors> element : m_df)
      sum += element.second.cpu->size() * sizeof(real);
    return sum;
  }
};

std::ostream& operator<<(std::ostream& os, DistributedDFGroup& df) {
  std::vector<Partition*> partitions = df.getPartitions();
  glm::ivec3 pMax = df.getNumPartitions();
  for (int q = 0; q < df.getQ(); q++) {
    for (int pz = 0; pz < pMax.z; pz++) {
      for (int py = 0; py < pMax.y; py++) {
        for (int px = 0; px < pMax.x; px++) {
          Partition* partition = df.getPartition(px, py, pz);
          os << "q=" << q << ", partition=" << glm::ivec3(px, py, pz)
             << std::endl;

          glm::ivec3 min = partition->getLatticeMin() - glm::ivec3(1, 1, 1);
          glm::ivec3 max = partition->getLatticeMax() + glm::ivec3(1, 1, 1);
          for (int z = min.z; z < max.z; z++) {
            for (int y = min.y; y < max.y; y++) {
              for (int x = min.x; x < max.x; x++) {
                try {
                  os << df(*partition, q, x, y, z);
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
      }
    }
  }
  return os;
}
