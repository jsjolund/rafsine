#pragma once

#include <thrust/copy.h>
#include <thrust/device_vector.h>
#include <thrust/fill.h>
#include <thrust/generate.h>
#include <thrust/system/cuda/execution_policy.h>
#include <thrust/transform_reduce.h>

#include <algorithm>
#include <iomanip>
#include <iostream>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "CudaUtils.hpp"
#include "DistributedLattice.hpp"

enum MemoryType { HOST_MEMORY, DEVICE_MEMORY };

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
  std::unordered_map<Partition, MemoryStore*> m_arrays;

  void memcpy3DAsync(const DistributionArray<T>& src, Partition srcPart,
                     int srcQ, glm::ivec3 srcPos, glm::ivec3 srcDim,
                     DistributionArray<T>* dst, Partition dstPart, int dstQ,
                     glm::ivec3 dstPos, glm::ivec3 dstDim, glm::ivec3 cpyExt,
                     cudaStream_t stream) {
    cudaMemcpy3DParms cpy = {0};
    // Source pointer
    cpy.srcPtr = make_cudaPitchedPtr(
        src.gpu_ptr(srcPart, srcQ, srcPos.x, srcPos.y, srcPos.z),
        srcDim.x * sizeof(T), srcDim.x, srcDim.y);
    // Destination pointer
    cpy.dstPtr = make_cudaPitchedPtr(
        dst->gpu_ptr(dstPart, dstQ, dstPos.x, dstPos.y, dstPos.z),
        dstDim.x * sizeof(T), dstDim.x, dstDim.y);
    // Extent of 3D copy
    cpy.extent = make_cudaExtent(cpyExt.x * sizeof(T), cpyExt.y, cpyExt.z);
    cpy.kind = cudaMemcpyDefault;

    CUDA_RT_CALL(cudaMemcpy3DAsync(&cpy, stream));
  }

 public:
  void deallocate(MemoryType type, Partition p = Partition());

  thrust::device_vector<T>* getDeviceVector(Partition partition) {
    if (m_arrays.find(partition) == m_arrays.end())
      throw std::out_of_range("Partition not allocated");
    return m_arrays[partition]->gpu;
  }
  thrust::host_vector<T>* getHostVector(Partition partition) {
    if (m_arrays.find(partition) == m_arrays.end())
      throw std::out_of_range("Partition not allocated");
    return m_arrays[partition]->cpu;
  }

  struct division : public thrust::unary_function<T, T> {
    const T m_arg;
    __host__ __device__ T operator()(const T& x) const { return x / m_arg; }
    explicit division(T arg) : m_arg(arg) {}
  };

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

  void allocate(Partition p = Partition());

  inline bool isAllocated(Partition p) const {
    return m_arrays.find(p) != m_arrays.end();
  }

  std::vector<Partition> getAllocatedPartitions();

  // Fill the ith array, i.e. the ith distribution function with a constant
  // value for all nodes
  void fill(T value, cudaStream_t stream = 0);

  // Read/write to specific allocated partition, including halos
  // start at -1 end at n + 1
  T& operator()(Partition partition, unsigned int q, int x, int y, int z = 0);

  T read(Partition partition, unsigned int q, int x, int y, int z = 0) const;

  // Return a pointer to the beginning of the GPU memory
  T* gpu_ptr(Partition partition, unsigned int q = 0, int x = 0, int y = 0,
             int z = 0) const;

  // Upload the distributions functions from the CPU to the GPU
  DistributionArray& upload();

  // Download the distributions functions from the GPU to the CPU
  DistributionArray& download();

  void gather(Partition srcPart, DistributionArray* dst,
              cudaStream_t stream = 0);
  void gather(int srcQ, int dstQ, Partition srcPart, DistributionArray<T>* dst,
              cudaStream_t stream = 0);
  void gather(glm::ivec3 globalMin, glm::ivec3 globalMax, int srcQ, int dstQ,
              Partition srcPart, DistributionArray<T>* dst, Partition dstPart,
              cudaStream_t stream = 0);
  void gatherSlice(glm::ivec3 slicePos, int srcQ, int dstQ, Partition srcPart,
                   DistributionArray<T>* dst, cudaStream_t stream = 0);

  void scatter(const DistributionArray& src, Partition dstPart,
               cudaStream_t stream = 0);

  // Static function to swap two DistributionArraysGroup
  static void swap(DistributionArray* f1, DistributionArray* f2);

  void exchange(Partition partition, DistributionArray* ndf,
                Partition neighbour, D3Q7::Enum direction,
                cudaStream_t stream = 0);

  size_t size(Partition partition) { return m_arrays[partition]->gpu->size(); }

  T getMin(Partition partition) const;
  T getMax(Partition partition) const;
  T getAverage(Partition partition, unsigned int q, unsigned int offset,
               unsigned int length, T divisor);

  friend std::ostream& operator<<(std::ostream& os,
                                  DistributionArray<T> const& df) {
    std::vector<Partition> partitions = df.getPartitions();
    glm::ivec3 numSubLats = df.getNumPartitions();
    for (int q = 0; q < df.getQ(); q++) {
      for (int pz = 0; pz < numSubLats.z; pz++) {
        for (int py = 0; py < numSubLats.y; py++) {
          for (int px = 0; px < numSubLats.x; px++) {
            Partition partition = df.getPartition(px, py, pz);

            if (!df.isAllocated(partition)) continue;

            os << "q=" << q << ", partition=" << glm::ivec3(px, py, pz)
               << std::endl;

            glm::ivec3 min(0, 0, 0);
            glm::ivec3 max = partition.getDims() + partition.getHalo() * 2;

            for (int z = min.z; z < max.z; z++) {
              for (int y = min.y; y < max.y; y++) {
                for (int x = min.x; x < max.x; x++) {
                  try {
                    os << std::setfill('0') << std::setw(1)
                       << df.read(partition, q, x, y, z);
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
