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
  //! Voxels is a fluid type
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

class VoxelArray : public DistributionArray<voxel_t> {
 public:
  VoxelArray(unsigned int nx,
             unsigned int ny,
             unsigned int nz,
             unsigned int nd,
             D3Q4::Enum partitioning)
      : DistributionArray<voxel_t>(1, nx, ny, nz, nd, 0, partitioning) {}

  inline voxel_t& operator()(int x, int y, int z) {
    return DistributionArray::operator()(getAllocatedPartitions().at(0), 0, x,
                                         y, z);
  }

  inline voxel_t* gpu_ptr() {
    return DistributionArray::gpu_ptr(getAllocatedPartitions().at(0), 0, 0, 0,
                                      0);
  }

  inline voxel_t* gpu_ptr(Partition partition) {
    return DistributionArray::gpu_ptr(partition, 0, 0, 0, 0);
  }

  inline void allocate(Partition p = Partition()) {
    DistributionArray::allocate(p);
    fill(VoxelType::Enum::FLUID);
  }

  inline int getSizeX() const { return m_latticeSize.x(); }
  inline int getSizeY() const { return m_latticeSize.y(); }
  inline int getSizeZ() const { return m_latticeSize.z(); }

  inline voxel_t getVoxelReadOnly(unsigned int x,
                                  unsigned int y,
                                  unsigned int z) const {
    return read(getPartition(0, 0, 0), 0, x, y, z);
  }

  bool isEmpty(int x, int y, int z) const {
    bool outside = true;
    if (x < 0) return outside;
    if (y < 0) return outside;
    if (z < 0) return outside;
    if (x >= static_cast<int>(m_latticeSize.x())) return outside;
    if (y >= static_cast<int>(m_latticeSize.y())) return outside;
    if (z >= static_cast<int>(m_latticeSize.z())) return outside;
    voxel_t data = getVoxelReadOnly(x, y, z);
    return ((data == VoxelType::Enum::EMPTY) ||
            (data == VoxelType::Enum::FLUID));
  }

  bool isEmptyStrict(int x, int y, int z) const {
    bool outside = true;
    if (x < 0) return outside;
    if (y < 0) return outside;
    if (z < 0) return outside;
    if (x >= static_cast<int>(m_latticeSize.x())) return outside;
    if (y >= static_cast<int>(m_latticeSize.y())) return outside;
    if (z >= static_cast<int>(m_latticeSize.z())) return outside;
    voxel_t data = getVoxelReadOnly(x, y, z);
    return (data == VoxelType::Enum::EMPTY);
  }
};
