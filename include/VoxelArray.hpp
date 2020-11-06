#pragma once

#include "DistributionArray.hpp"

typedef unsigned int voxel_t;

/**
 * @brief Types of boundary conditions
 *
 */
namespace VoxelType {
/**
 * @brief Enumerated boundary condition types
 *
 */
enum Enum {
  //! Voxel is a fluid type
  FLUID = 0,
  //! Voxel is ignored in simulation
  EMPTY = 1,
  //! Half-way bounce-back boundary condition
  WALL = 2,
  //! TODO(Unused)
  FREE_SLIP = 3,
  //! Voxel has a constant temperature and velocity output
  INLET_CONSTANT = 4,
  //! Sets the temperature gradient to zero
  INLET_ZERO_GRADIENT = 5,
  //! Integrates the temperature at a relative position and adds it to the
  //! lattice site with optional time constants
  INLET_RELATIVE = 6
};
}  // namespace VoxelType

/**
 * @brief Voxel array distributed across multiple GPUs. Holds an integer for
 * each lattice site representing its boundary condition ID. The boundary
 * condition definition is in a separate array.
 */
class VoxelArray : public DistributionArray<voxel_t> {
 public:
  /**
   * @brief Construct a new voxel array
   *
   * @param nx Size on X-axis
   * @param ny Size on Y-axis
   * @param nz Size on Z-axis
   * @param nd Number of CUDA devices
   * @param partitioning Axes to partition along
   */
  VoxelArray(unsigned int nx,
             unsigned int ny,
             unsigned int nz,
             unsigned int nd,
             D3Q4::Enum partitioning)
      : DistributionArray<voxel_t>(1, nx, ny, nz, nd, 0, partitioning) {}

  /**
   * @brief Read/write to a voxel
   *
   * @param x
   * @param y
   * @param z
   * @return voxel_t&
   */
  inline voxel_t& operator()(int x, int y, int z) {
    return DistributionArray::operator()(getAllocatedPartitions().at(0), 0, x,
                                         y, z);
  }

  /**
   * @brief Pointer to beginning of GPU array assuming only one partition
   *
   * @return voxel_t*
   */
  inline voxel_t* gpu_ptr() {
    return DistributionArray::gpu_ptr(getAllocatedPartitions().at(0), 0, 0, 0,
                                      0);
  }

  /**
   * @brief Pointer to beginning of GPU array on specified partition
   *
   * @param partition
   * @return voxel_t*
   */
  inline voxel_t* gpu_ptr(Partition partition) {
    return DistributionArray::gpu_ptr(partition, 0, 0, 0, 0);
  }

  /**
   * @brief Allocate a partition and fill it with the fluid boundary condition
   *
   * @param p
   */
  inline void allocate(Partition p = Partition()) {
    DistributionArray::allocate(p);
    fill(VoxelType::Enum::FLUID);
  }

  /**
   * @return size_t Size along X-axis
   */
  inline size_t getSizeX() const { return m_latticeSize.x(); }
  /**
   * @return size_t Size along Y-axis
   */
  inline size_t getSizeY() const { return m_latticeSize.y(); }
  /**
   * @return size_t Size along Z-axis
   */
  inline size_t getSizeZ() const { return m_latticeSize.z(); }

  /**
   * @brief Read (only) voxel ID at position
   *
   * @param x
   * @param y
   * @param z
   * @return voxel_t
   */
  inline voxel_t getVoxelReadOnly(unsigned int x,
                                  unsigned int y,
                                  unsigned int z) const {
    return read(getPartition(0, 0, 0), 0, x, y, z);
  }

  /**
   * @brief True if voxel is empty or fluid type, false otherwise
   *
   * @param x
   * @param y
   * @param z
   * @return true
   * @return false
   */
  bool isEmpty(int x, int y, int z) const {
    if (x < 0) return true;
    if (y < 0) return true;
    if (z < 0) return true;
    if (x >= static_cast<int>(m_latticeSize.x())) return true;
    if (y >= static_cast<int>(m_latticeSize.y())) return true;
    if (z >= static_cast<int>(m_latticeSize.z())) return true;
    voxel_t data = getVoxelReadOnly(x, y, z);
    return ((data == VoxelType::Enum::EMPTY) ||
            (data == VoxelType::Enum::FLUID));
  }

  /**
   * @brief True if voxel is empty, false otherwise
   *
   * @param x
   * @param y
   * @param z
   * @return true
   * @return false
   */
  bool isEmptyStrict(int x, int y, int z) const {
    if (x < 0) return true;
    if (y < 0) return true;
    if (z < 0) return true;
    if (x >= static_cast<int>(m_latticeSize.x())) return true;
    if (y >= static_cast<int>(m_latticeSize.y())) return true;
    if (z >= static_cast<int>(m_latticeSize.z())) return true;
    voxel_t data = getVoxelReadOnly(x, y, z);
    return (data == VoxelType::Enum::EMPTY);
  }
};
