#pragma once

#include <thrust/copy.h>
#include <thrust/device_vector.h>
#include <thrust/extrema.h>
#include <thrust/fill.h>
#include <thrust/generate.h>
#include <thrust/remove.h>
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
#include "Vector3.hpp"

enum MemoryType { HOST_MEMORY, DEVICE_MEMORY };

/**
 * @brief An array for lattices distributed across multiple GPUs
 *
 * @tparam T Numeric type
 */
template <typename T>
class DistributionArray : public DistributedLattice {
 protected:
  /**
   * @brief Thrust GPU and CPU vector storage
   */
  struct MemoryStore {
    thrust::device_vector<T>* gpu;
    thrust::host_vector<T>* cpu;
    explicit MemoryStore(size_t size);
  };
  const unsigned int m_Q;
  std::unordered_map<Partition, MemoryStore*> m_arrays;

  /**
   * @brief Perform asynchronous copy of 3D volume between GPUs using
   * multistreaming
   *
   * @param src Source distribution array
   * @param srcPart Source partition in distribution array
   * @param srcQ 4D coordinate (i.e. lattice direction)
   * @param srcPos 3D position in source array (i.e. with or without ghost
   * layers)
   * @param srcDim Extents of source partition
   * @param dst Destination distribution array
   * @param dstPart Destination partition in distribution array
   * @param dstQ 4D coordinate (i.e. lattice direction)
   * @param dstPos 3D position in destination array
   * @param dstDim Extents of destination partition
   * @param cpyExt Extents of copy
   * @param stream Cuda stream to use
   */
  void memcpy3DAsync(const DistributionArray<T>& src,
                     Partition srcPart,
                     unsigned int srcQ,
                     Vector3<unsigned int> srcPos,
                     Vector3<size_t> srcDim,
                     DistributionArray<T>* dst,
                     Partition dstPart,
                     unsigned int dstQ,
                     Vector3<unsigned int> dstPos,
                     Vector3<size_t> dstDim,
                     Vector3<size_t> cpyExt,
                     cudaStream_t stream);

 public:
  /**
   * @brief Deallocate partition on host or device
   *
   * @param type Host or device
   * @param p Partition to deallocate
   */
  void deallocate(MemoryType type, Partition p = Partition());
  /**
   * @brief Pointer to Thrust device vector
   *
   * @param partition The partition in distribution array
   * @return thrust::device_vector<T>* The pointer
   */
  thrust::device_vector<T>* getDeviceVector(Partition partition) {
    if (m_arrays.find(partition) == m_arrays.end())
      throw std::out_of_range("Partition not allocated");
    return m_arrays[partition]->gpu;
  }
  /**
   * @brief Pointer to Thrust host vector
   *
   * @param partition The partition in distribution array
   * @return thrust::host_vector<T>* The pointer
   */
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
   * optionally create an extra "ghostLayer" array around the partitions and
   * stream data between adjacent partitions.
   *
   * @param q Number of 3D arrays
   * @param nx Size of 3D X-axis
   * @param ny Size of 3D Y-axis
   * @param nz Size of 3D Z-axis
   * @param nd Number of devices
   * @param ghostLayerSize Size of ghostLayer along decomposition axis
   */
  DistributionArray(unsigned int q,
                    unsigned int nx,
                    unsigned int ny,
                    unsigned int nz,
                    unsigned int nd,
                    unsigned int ghostLayerSize,
                    D3Q4::Enum partitioning);

  ~DistributionArray();

  DistributionArray& operator=(const DistributionArray& f);

  /**
   * @return unsigned int Size of 4:th dimension (i.e. lattice direction)
   */
  inline unsigned int getQ() const { return m_Q; }

  /**
   * @brief Allocate partition on both host and device memory
   *
   * @param p The partition to allocate
   */
  void allocate(Partition p = Partition());
  /**
   * @brief Check if partition is allocated
   *
   * @param p  The partition
   * @return true
   * @return false
   */
  inline bool isAllocated(Partition p) const {
    return m_arrays.find(p) != m_arrays.end();
  }
  /**
   * @return std::vector<Partition> List of all allocated partitions
   */
  std::vector<Partition> getAllocatedPartitions();

  /**
   * @brief Fill the ith array, i.e. the ith distribution function with a
   * constant value for all nodes
   *
   * @param value
   * @param stream
   */
  void fill(T value, cudaStream_t stream = 0);

  /**
   * @brief Read/write to specific allocated partition, including ghost layers
   * (starts at -1, end at n + 1)
   *
   * @param partition
   * @param q
   * @param x
   * @param y
   * @param z
   * @return T&
   */
  T& operator()(Partition partition, unsigned int q, int x, int y, int z = 0);

  /**
   * @brief Read (only) to specific allocated partition, including ghost layers
   * (starts at -1, end at n + 1)
   *
   * @param partition
   * @param q
   * @param x
   * @param y
   * @param z
   * @return T
   */
  T read(Partition partition, unsigned int q, int x, int y, int z = 0) const;

  /**
   * @brief Return a pointer to the beginning of the GPU memory
   *
   * @param partition
   * @param q
   * @param x
   * @param y
   * @param z
   * @return T*
   */
  T* gpu_ptr(Partition partition,
             unsigned int q = 0,
             int x = 0,
             int y = 0,
             int z = 0) const;

  /**
   * @brief Upload the distributions functions from the CPU to the GPU
   *
   * @return DistributionArray&
   */
  DistributionArray& upload();

  /**
   * @brief Download the distributions functions from the GPU to the CPU
   *
   * @return DistributionArray&
   */
  DistributionArray& download();

  /**
   * @brief Gather all values from this distribution array into another.
   * Lattices must have the same dimensions and extents.
   *
   * @param srcPart
   * @param dst
   * @param stream
   */
  void gather(Partition srcPart,
              DistributionArray* dst,
              cudaStream_t stream = 0);

  /**
   * @brief Gather values all values from this distribution array to another for
   * specified 4:th dimension (i.e. lattice direction).  Lattices must have the
   * same dimensions and extents.
   *
   * @param srcQ
   * @param dstQ
   * @param srcPart
   * @param dst
   * @param stream
   */
  void gather(unsigned int srcQ,
              unsigned int dstQ,
              Partition srcPart,
              DistributionArray<T>* dst,
              cudaStream_t stream = 0);

  /**
   * @brief Gather all values from this distribution array to another for
   * specified 3D volume and 4:th dimension.
   *
   * @param globalMin
   * @param globalMax
   * @param srcQ
   * @param dstQ
   * @param srcPart
   * @param dst
   * @param dstPart
   * @param stream
   */
  void gather(Vector3<unsigned int> globalMin,
              Vector3<unsigned int> globalMax,
              unsigned int srcQ,
              unsigned int dstQ,
              Partition srcPart,
              DistributionArray<T>* dst,
              Partition dstPart,
              cudaStream_t stream = 0);

  /**
   * @brief Gather a slice (2D quad) between distribution arrays
   *
   * @param slicePos
   * @param srcQ
   * @param dstQ
   * @param srcPart
   * @param dst
   * @param stream
   */
  void gatherSlice(Vector3<unsigned int> slicePos,
                   unsigned int srcQ,
                   unsigned int dstQ,
                   Partition srcPart,
                   DistributionArray<T>* dst,
                   cudaStream_t stream = 0);

  /**
   * @brief Scatter values from source distribution array into this array
   *
   * @param src
   * @param dstPart
   * @param stream
   */
  void scatter(const DistributionArray& src,
               Partition dstPart,
               cudaStream_t stream = 0);

  /**
   * @brief Static function to swap two DistributionArrays
   *
   * @param f1
   * @param f2
   */
  static void swap(DistributionArray* f1, DistributionArray* f2);

  /**
   * @brief Exchange ghost layers along direction between parititions
   *
   * @param partition
   * @param ndf
   * @param neighbour
   * @param direction
   * @param stream
   */
  void exchange(Partition partition,
                DistributionArray* ndf,
                Partition neighbour,
                D3Q7::Enum direction,
                cudaStream_t stream = 0);

  /**
   * @brief Total size of partition on GPU
   *
   * @param partition
   * @return size_t
   */
  size_t size(Partition partition) { return m_arrays[partition]->gpu->size(); }

  /**
   * @brief Get minimum value in array
   *
   * @param partition
   * @return T
   */
  T getMin(Partition partition) const;

  /**
   * @brief Get maximum value in array
   *
   * @param partition
   * @return T
   */
  T getMax(Partition partition) const;

  /**
   * @brief Get average value over range in partition
   *
   * @param partition The partition
   * @param q 4:th dimension coordinate
   * @param offset Offset into array
   * @param length Length of range
   * @param divisor
   * @return T
   */
  T getAverage(Partition partition,
               unsigned int q,
               unsigned int offset,
               unsigned int length,
               T divisor);

  friend std::ostream& operator<<(std::ostream& os,
                                  DistributionArray<T> const& df) {
    std::vector<Partition> partitions = df.getPartitions();
    Vector3<int> numSubLats = df.getNumPartitions();
    for (int q = 0; q < df.getQ(); q++) {
      for (int pz = 0; pz < numSubLats.z(); pz++) {
        for (int py = 0; py < numSubLats.y(); py++) {
          for (int px = 0; px < numSubLats.x(); px++) {
            Partition partition = df.getPartition(px, py, pz);

            if (!df.isAllocated(partition)) continue;

            os << "q=" << q << ", partition=" << Vector3<int>(px, py, pz)
               << std::endl;

            Vector3<int> min(0, 0, 0);
            Vector3<size_t> max =
                partition.getExtents() + partition.getGhostLayer() * (size_t)2;

            for (int z = min.z(); z < max.z(); z++) {
              for (int y = min.y(); y < max.y(); y++) {
                for (int x = min.x(); x < max.x(); x++) {
                  try {
                    os << std::setfill('0') << std::setw(1)
                       << df.read(partition, q, x, y, z);
                  } catch (std::out_of_range& e) { os << "X"; }
                  if (x < max.x() - 1) os << ",";
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

template __global__ class DistributionArray<real_t>;
template __global__ class DistributionArray<unsigned int>;
