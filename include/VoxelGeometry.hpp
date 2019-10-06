#pragma once

#include <algorithm>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "BoundaryCondition.hpp"
#include "CudaUtils.hpp"
#include "ErrorFormat.hpp"
#include "UnitConverter.hpp"
#include "VoxelArray.hpp"
#include "VoxelObject.hpp"

#define DEFAULT_GEOMETRY_NAME "geometry"

// Class to generate the array of voxels from quads and boxes
class VoxelGeometry {
 private:
  voxel_t m_voxelTypeCounter;
  std::unordered_map<size_t, BoundaryCondition> m_types;

 protected:
  //! Sizes of lattice
  const int m_nx, m_ny, m_nz;

  //! GPU distributable array of voxels / boundary condition ids
  std::shared_ptr<VoxelArray> m_voxelArray;

  //! Array of boundary conditions stored by their id
  std::shared_ptr<BoundaryConditions> m_bcsArray;

  //! Volumes of voxels where simulation time averaged values should be read
  std::shared_ptr<VoxelVolumeArray> m_sensorArray;

  //! Hashmap with key boundary condition id, value geometry name combined with
  //! this id
  std::unordered_map<voxel_t,
                     std::unordered_set<std::string, std::hash<std::string>>>
      m_voxNameMap;

  //! Hashmap with key geometry name, value all voxel quads sharing this name
  std::unordered_map<std::string,
                     std::unordered_set<VoxelQuad, std::hash<VoxelQuad>>>
      m_nameQuadMap;

  /**
   * @brief Stores a pair of boundary condition struct and geometry name string
   * in a hashmap with key as their combined hash, and value the boundary
   * condition. When a name and struct combo is not found in the map, a new
   * boundary condition is stored and id counter incremented. Otherwise the id
   * of existing combo is returned.
   *
   * @param bc
   * @param geoName
   * @return voxel_t
   */
  voxel_t storeType(BoundaryCondition *bc, const std::string &geoName);

  //! Set a position in the voxel array to a voxel id
  inline void set(unsigned int x, unsigned int y, unsigned int z,
                  voxel_t value) {
    (*m_voxelArray)(x - 1, y - 1, z - 1) = value;
  }

  //! Set a position in the voxel array to a voxel id
  inline void set(Eigen::Vector3i v, voxel_t value) {
    set(v.x(), v.y(), v.z(), value);
  }

 public:
  std::unordered_map<Eigen::Vector3i, std::string> getLabels();

  inline std::shared_ptr<VoxelArray> getVoxelArray() { return m_voxelArray; }

  inline std::shared_ptr<BoundaryConditions> getBoundaryConditions() {
    return m_bcsArray;
  }

  inline std::unordered_set<std::string> getObjectNamesById(voxel_t id) {
    return m_voxNameMap.at(id);
  }

  inline std::unordered_set<VoxelQuad> getQuadsByName(std::string name) {
    return m_nameQuadMap.at(name);
  }

  std::unordered_set<voxel_t> getVoxelsByName(std::string name);

  std::vector<std::string> getGeometryNames();

  inline std::shared_ptr<VoxelVolumeArray> getSensors() {
    return m_sensorArray;
  }

  inline int getNumTypes() { return m_voxelTypeCounter; }

  inline voxel_t get(unsigned int x, unsigned int y, unsigned int z) {
    return (*m_voxelArray)(x - 1, y - 1, z - 1);
  }

  voxel_t inline get(Eigen::Vector3i v) { return get(v.x(), v.y(), v.z()); }

  inline int getNx() { return m_nx; }

  inline int getNy() { return m_ny; }

  inline int getNz() { return m_nz; }

  inline int getSize() { return m_nx * m_ny * m_nz; }

  ~VoxelGeometry() {}

  VoxelGeometry();

  VoxelGeometry(const int nx, const int ny, const int nz);
};

std::ostream &operator<<(std::ostream &Str, VoxelGeometry &v);
