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
#include "Lattice.hpp"

// Define a group of array on the GPU
// designed for 3D arrays (but works in 2D as well)
// Useful to group all the distribution functions into a single array
// distribution functions (fi) are packed in memory based on their direction:
// memory: f1,f1,...,f1,f2,f2,...,f2,f3,f3,...
class DistributionFunction : public Lattice {
 private:
  struct thrust_vectors {
    thrust::device_vector<real>* gpu;
    thrust::host_vector<real>* cpu;
  };

  // Distribution functions on

 public:
  std::unordered_map<SubLattice, thrust_vectors> m_df;
  // Constructor
  DistributionFunction(unsigned int Q, unsigned int latticeSizeX,
                       unsigned int latticeSizeY, unsigned int latticeSizeZ,
                       unsigned int subdivisions = 0);

  ~DistributionFunction();

  DistributionFunction& operator=(const DistributionFunction& f);

  void allocate(SubLattice p);
  inline bool isAllocated(SubLattice p) { return m_df.find(p) != m_df.end(); }

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
  real* gpu_ptr(SubLattice subLattice, unsigned int dfIdx, int x, int y, int z,
                bool halo = false);
  real* gpu_ptr(SubLattice subLattice, unsigned int dfIdx = 0);

  void pushSubLattice(int srcDev, SubLattice subLattice, int dstDev,
                      DistributionFunction* nDf, cudaStream_t cpyStream);
  // Upload the distributions functions from the CPU to the GPU
  DistributionFunction& upload();
  // Download the distributions functions from the GPU to the CPU
  DistributionFunction& download();

  // Static function to swap two DistributionFunctionsGroup
  static void swap(DistributionFunction* f1, DistributionFunction* f2);

  unsigned long memoryUse();
};

std::ostream& operator<<(std::ostream& os, DistributionFunction& df);
