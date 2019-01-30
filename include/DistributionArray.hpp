#pragma once

#include <thrust/copy.h>
#include <thrust/device_vector.h>
#include <thrust/fill.h>
#include <thrust/generate.h>

#include <algorithm>
#include <iomanip>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

#include "CudaUtils.hpp"
#include "DistributedLattice.hpp"

// Define a group of array on the GPU
// designed for 3D arrays (but works in 2D as well)
// Useful to group all the distribution functions into a single array
// distribution functions (fi) are packed in memory based on their direction:
// memory: f1,f1,...,f1,f2,f2,...,f2,f3,f3,...
class DistributionArray : public DistributedLattice {
 protected:
  struct thrust_vectors {
    thrust::device_vector<real>* gpu;
    thrust::host_vector<real>* cpu;
  };
  unsigned int m_Q;
  std::unordered_map<SubLattice, thrust_vectors> m_arrays;

 public:
  // Constructor
  DistributionArray(unsigned int Q, unsigned int latticeSizeX,
                    unsigned int latticeSizeY, unsigned int latticeSizeZ,
                    unsigned int subdivisions = 1, unsigned int haloSize = 0);

  ~DistributionArray();

  DistributionArray& operator=(const DistributionArray& f);

  inline unsigned int getQ() { return m_Q; }

  void allocate(SubLattice p);

  inline bool isAllocated(SubLattice p) {
    return m_arrays.find(p) != m_arrays.end();
  }

  std::vector<SubLattice> getAllocatedSubLattices();

  // Fill the ith array, i.e. the ith distribution function with a constant
  // value for all nodes
  void fill(unsigned int dfIdx, real value);

  // Read/write to allocated subLattices, excluding halos
  real& operator()(unsigned int dfIdx, unsigned int x, unsigned int y,
                   unsigned int z = 0);

  // Read/write to specific allocated subLattice, including halos
  // start at -1 end at n + 1
  real& operator()(SubLattice subLattice, unsigned int dfIdx, int x, int y,
                   int z = 0);

  // Return a pointer to the beginning of the GPU memory
  real* gpu_ptr(SubLattice subLattice, unsigned int dfIdx = 0, int x = 0,
                int y = 0, int z = 0);

  void pushSubLattice(int srcDev, SubLattice subLattice, int dstDev,
                      DistributionArray* nDf, cudaStream_t cpyStream);

  // Upload the distributions functions from the CPU to the GPU
  DistributionArray& upload();

  // Download the distributions functions from the GPU to the CPU
  DistributionArray& download();

  // Static function to swap two DistributionArraysGroup
  static void swap(DistributionArray* f1, DistributionArray* f2);

  void haloExchange(SubLattice subLattice, DistributionArray* ndf,
                    SubLattice neighbour, D3Q7::Enum direction,
                    cudaStream_t stream);

  unsigned long memoryUse();
};

std::ostream& operator<<(std::ostream& os, DistributionArray& df);
