/**
 * \file Voxel.hpp
 * \brief Define an array of voxel class
 * \author Nicolas Delbosc
 * \date June 2014
 *****************************************************************************/
#pragma once

#include <algorithm>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#include "BoundaryCondition.hpp"
#include "Primitives.hpp"

/**
 * @brief Store a 3D array of voxels
 *
 */
class VoxelArray {
 private:
  //! Size of the domain
  unsigned int m_sizeX, m_sizeY, m_sizeZ;
  //! Pointer to the data on the cpu
  voxel *m_data;
  //! Data on the GPU
  voxel *m_data_d;

 public:
  //! Constructor
  VoxelArray(unsigned int sizeX, unsigned int sizeY, unsigned int sizeZ)
      : m_sizeX(sizeX), m_sizeY(sizeY), m_sizeZ(sizeZ) {
    m_data = new voxel[getFullSize()];
    memset(m_data, 0, getFullSize() * sizeof(voxel));
    CUDA_RT_CALL(cudaMalloc(reinterpret_cast<void **>(&m_data_d),
                            sizeof(voxel) * getFullSize()));
  }
  //! Copy constructor
  VoxelArray(const VoxelArray &other)
      : m_sizeX(other.m_sizeX), m_sizeY(other.m_sizeY), m_sizeZ(other.m_sizeZ) {
    // m_data = other.m_data;
    m_data = new voxel[other.getFullSize()];
    memcpy(m_data, other.m_data, sizeof(voxel) * getFullSize());
    CUDA_RT_CALL(cudaMalloc(reinterpret_cast<void **>(&m_data_d),
                            sizeof(voxel) * getFullSize()));
  }
  //! Assignment operator
  VoxelArray &operator=(const VoxelArray &other);

  //! Destructor
  ~VoxelArray() {
    CUDA_RT_CALL(cudaFree(m_data_d));
    delete[] m_data;
  }
  inline unsigned int getSizeX() const { return m_sizeX; }
  inline unsigned int getSizeY() const { return m_sizeY; }
  inline unsigned int getSizeZ() const { return m_sizeZ; }
  inline unsigned int getFullSize() const {
    return m_sizeX * m_sizeY * m_sizeZ;
  }
  //! Provide easy read and write access to voxels
  inline voxel &operator()(unsigned int x, unsigned int y, unsigned int z) {
    return m_data[x + y * m_sizeX + z * m_sizeX * m_sizeY];
  }
  //! Provide read-only access
  inline voxel getVoxelReadOnly(unsigned int x, unsigned int y,
                                unsigned int z) const {
    return m_data[x + y * m_sizeX + z * m_sizeX * m_sizeY];
  }
  //! Send the data to the GPU
  inline void upload() const {
    CUDA_RT_CALL(cudaMemcpy(m_data_d, m_data, sizeof(voxel) * getFullSize(),
                            cudaMemcpyHostToDevice));
  }
  //! Returns a pointer to the gpu data
  inline voxel *gpu_ptr() { return m_data_d; }

  /**
   * @brief Check that the coordinates are inside the domain and if it is empty
   * (outside domain) or a fluid
   *
   * @param x
   * @param y
   * @param z
   * @return true If the voxel is of type VOX_EMPTY or VOX_FLUID
   * @return false Otherwises
   */
  bool isEmpty(unsigned int x, unsigned int y, unsigned int z) const;

  /**
   * @brief Convenience function for isEmpty
   *
   * @param position
   * @return true
   * @return false
   */
  inline bool isEmpty(vec3ui position) const {
    return isEmpty(position.x, position.y, position.z);
  }

  /**
   * @brief Check that the coordinates are inside the domain and if it is empty
   * (outside domain)
   *
   * @param x
   * @param y
   * @param z
   * @return true If the voxel is of type VOX_EMPTY
   * @return false Otherwise
   */
  bool isEmptyStrict(unsigned int x, unsigned int y, unsigned int z) const;

  /**
   * @brief Save the voxels to file
   *
   * @param filename
   */
  void saveToFile(std::string filename);

  /**
   * @brief Save only the chunk of voxels that are not empty
   *
   * @param filename
   */
  void saveAutocrop(std::string filename);

  /**
   * @brief Load the voxels from a file
   *
   * @param filename
   */
  void loadFromFile(std::string filename);

  /**
   * @brief Fill the whole array with a unique value
   *
   * @param value
   */
  inline void fill(voxel value) {
    for (unsigned int i = 0; i < getFullSize(); i++) m_data[i] = value;
  }
};
