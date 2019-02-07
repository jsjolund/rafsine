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

template <class T>
class DistributionArray : public DistributedLattice {
 protected:
  struct MemoryStore {
    thrust::device_vector<T>* gpu;
    thrust::host_vector<T>* cpu;
    explicit MemoryStore(size_t size) {
      gpu = new thrust::device_vector<T>(size);
      cpu = new thrust::host_vector<T>(size);
    }
  };
  const unsigned int m_Q;
  std::unordered_map<SubLattice, MemoryStore*> m_arrays;

 public:
  /**
   * @brief A 4D array decomposed into windows/partitions by 3D, 2D or 1D
   * divisions. Used for storing arrays distributed on multiple GPUs. Can
   * optionally create an extra "halo" array around the partitions and stream
   * data between adjacent partitions.
   *
   * @param q Number of 3D arrays
   * @param nx Size of 3D X-axis
   * @param ny Size of 3D Y-axis
   * @param nz Size of 3D Z-axis
   * @param nd Number of devices
   * @param haloSize Size of halo along decomposition axis
   */
  DistributionArray(unsigned int q, unsigned int nx, unsigned int ny,
                    unsigned int nz, unsigned int nd = 1,
                    unsigned int haloSize = 0);

  ~DistributionArray();

  DistributionArray& operator=(const DistributionArray& f);

  inline unsigned int getQ() const { return m_Q; }

  void allocate(SubLattice p = SubLattice());

  inline bool isAllocated(SubLattice p) {
    return m_arrays.find(p) != m_arrays.end();
  }

  std::vector<SubLattice> getAllocatedSubLattices();

  // Fill the ith array, i.e. the ith distribution function with a constant
  // value for all nodes
  void fill(unsigned int q, T value);

  // Read/write to specific allocated subLattice, including halos
  // start at -1 end at n + 1
  T& operator()(SubLattice subLattice, unsigned int q, int x, int y, int z = 0);

  // Return a pointer to the beginning of the GPU memory
  T* gpu_ptr(SubLattice subLattice, unsigned int q = 0, int x = 0, int y = 0,
             int z = 0) const;

  // Upload the distributions functions from the CPU to the GPU
  DistributionArray& upload();

  // Download the distributions functions from the GPU to the CPU
  DistributionArray& download();

  void gather(SubLattice srcPart, DistributionArray* dst,
              cudaStream_t stream = 0);
  void gather(int srcQ, int dstQ, SubLattice srcPart, DistributionArray<T>* dst,
              cudaStream_t stream);

  void scatter(const DistributionArray& src, SubLattice dstPart,
               cudaStream_t stream = 0);

  // Static function to swap two DistributionArraysGroup
  static void swap(DistributionArray* f1, DistributionArray* f2);

  void exchange(SubLattice subLattice, DistributionArray* ndf,
                SubLattice neighbour, D3Q7::Enum direction,
                cudaStream_t stream);

  size_t size(SubLattice subLattice) {
    return m_arrays[subLattice].gpu->size();
  }

  void getMinMax(SubLattice subLattice, int* min, int* max);

  friend std::ostream& operator<<(std::ostream& os,
                                  DistributionArray<T> const& df) {
    std::vector<SubLattice> subLattices = df.getSubLattices();
    glm::ivec3 pMax = df.getNumSubLattices();
    for (int q = 0; q < df.getQ(); q++) {
      for (int pz = 0; pz < pMax.z; pz++) {
        for (int py = 0; py < pMax.y; py++) {
          for (int px = 0; px < pMax.x; px++) {
            SubLattice subLattice = df.getSubLattice(px, py, pz);

            if (!df.isAllocated(subLattice)) continue;

            os << "q=" << q << ", subLattice=" << glm::ivec3(px, py, pz)
               << std::endl;

            glm::ivec3 min = glm::ivec3(0, 0, 0);
            glm::ivec3 max =
                subLattice.getLatticeDims() + subLattice.getHalo() * 2;
            for (int z = max.z - 1; z >= min.z; z--) {
              for (int y = max.y - 1; y >= min.y; y--) {
                for (int x = min.x; x < max.x; x++) {
                  try {
                    os << std::setfill('0') << std::setw(2)
                       << df(subLattice, q, x, y, z);
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
};

#include "DistributionArrayImpl.hpp"