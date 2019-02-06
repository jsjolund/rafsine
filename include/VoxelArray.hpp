#pragma once

#include "DistributionArray.hpp"

typedef int voxel;

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
  EMPTY = -1,          //!< Voxel is ignored in simulation
  FLUID = 0,           //!< Voxels is a fluid type
  WALL = 1,            //!< Half-way bounce-back boundary condition
  FREE_SLIP = 2,       //!< TODO(Unused)
  INLET_CONSTANT = 3,  //!< Voxel has a constant temperature and velocity output
  INLET_ZERO_GRADIENT = 4,  //!< Sets the temperature gradient to zero
  INLET_RELATIVE = 5  //!< Integrates the temperature at a relative position and
                      // adds it to the voxel
};
}  // namespace VoxelType

class VoxelArray : public DistributionArray<voxel> {
 public:
  VoxelArray(unsigned int latticeSizeX, unsigned int latticeSizeY,
             unsigned int latticeSizeZ, unsigned int subdivisions = 1)
      : DistributionArray<voxel>(1, latticeSizeX, latticeSizeY, latticeSizeZ,
                                 subdivisions, 0) {
    if (subdivisions <= 1) allocate(getSubLattice(0, 0, 0));
  }

  inline voxel& operator()(int x, int y, int z) {
    return DistributionArray::operator()(getAllocatedSubLattices().at(0), 0, x,
                                         y, z);
  }

  inline voxel* gpu_ptr() {
    return DistributionArray::gpu_ptr(getAllocatedSubLattices().at(0), 0, 0, 0,
                                      0);
  }

  inline int getSizeX() const { return m_latticeSize.x; }
  inline int getSizeY() const { return m_latticeSize.y; }
  inline int getSizeZ() const { return m_latticeSize.z; }

  inline voxel getVoxelReadOnly(unsigned int x, unsigned int y,
                                unsigned int z) const {
    SubLattice subLattice = getSubLattice(0, 0, 0);
    thrust::host_vector<voxel>* cpuVec = m_arrays.at(subLattice)->cpu;
    glm::ivec3 srcLatDim = subLattice.getArrayDims();
    int idx = I4D(0, x, y, z, srcLatDim.x, srcLatDim.y, srcLatDim.z);
    return (*cpuVec)[idx];
  }

  bool isEmpty(int x, int y, int z) const {
    bool outside = true;
    if (x < 0) return outside;
    if (y < 0) return outside;
    if (z < 0) return outside;
    if (x >= static_cast<int>(m_latticeSize.x)) return outside;
    if (y >= static_cast<int>(m_latticeSize.y)) return outside;
    if (z >= static_cast<int>(m_latticeSize.z)) return outside;
    voxel data = getVoxelReadOnly(x, y, z);
    return ((data == VoxelType::Enum::EMPTY) ||
            (data == VoxelType::Enum::FLUID));
  }

  bool isEmptyStrict(int x, int y, int z) const {
    bool outside = true;
    if (x < 0) return outside;
    if (y < 0) return outside;
    if (z < 0) return outside;
    if (x >= static_cast<int>(m_latticeSize.x)) return outside;
    if (y >= static_cast<int>(m_latticeSize.y)) return outside;
    if (z >= static_cast<int>(m_latticeSize.z)) return outside;
    voxel data = getVoxelReadOnly(x, y, z);
    return (data == VoxelType::Enum::EMPTY);
  }
};
