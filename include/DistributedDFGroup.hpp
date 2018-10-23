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
                     unsigned int subdivisions = 0);

  ~DistributedDFGroup();

  DistributedDFGroup& operator=(const DistributedDFGroup& f);

  // Return the number of arrays in the group i.e. the number of distribution
  // functions
  unsigned int getQ() const { return m_Q; }

  void allocate(Partition p);

  std::vector<Partition> getAllocatedPartitions();

  // Fill the ith array, i.e. the ith distribution function with a constant
  // value for all nodes
  void fill(unsigned int df_idx, real value);

  // Read/write to allocated partitions, excluding halos
  real& operator()(unsigned int df_idx, unsigned int x, unsigned int y,
                   unsigned int z = 0);

  // Read/write to specific allocated partition, including halos
  // start at -1 end at n + 1
  real& operator()(Partition partition, unsigned int df_idx, int x, int y,
                   int z = 0);
  // Return a pointer to the beginning of the GPU memory
  real* gpu_ptr(Partition partition, unsigned int idx = 0);
  // Return a pointer to the beginning of the GPU memory
  real* gpu_ptr(Partition partition, unsigned int df_idx, int x, int y, int z);

  void pushHalo(int srcDev, Partition partition, int dstDev,
                DistributedDFGroup* nDf, HaloExchangeData haloData,
                cudaStream_t cpyStream);

  void pushPartition(int srcDev, Partition partition, int dstDev,
                     DistributedDFGroup* nDf, cudaStream_t cpyStream);
  // Upload the distributions functions from the CPU to the GPU
  DistributedDFGroup& upload();
  // Download the distributions functions from the GPU to the CPU
  DistributedDFGroup& download();

  // // Copy from another group of distribution functions
  // // SAME SIZE IS REQUIRED
  //  DistributedDFGroup &operator=(const DistributedDFGroup &f);
  // Static function to swap two DistributionFunctionsGroup
  static void swap(DistributedDFGroup* f1, DistributedDFGroup* f2);

  unsigned long memoryUse();
};

std::ostream& operator<<(std::ostream& os, DistributedDFGroup& df);
