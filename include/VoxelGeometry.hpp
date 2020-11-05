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
#include "Vector3.hpp"
#include "VoxelArray.hpp"
#include "VoxelObject.hpp"

#define DEFAULT_GEOMETRY_NAME "geometry"

// Class to generate the array of voxels from quads and boxes
class VoxelGeometry {
 private:
  voxel_t m_voxelTypeCounter;
  std::unordered_map<size_t, BoundaryCondition> m_types;

 protected:
  int m_incompatible;

  //! GPU distributable array of voxels / boundary condition ids
  std::shared_ptr<VoxelArray> m_voxelArray;

  //! Array of boundary conditions stored by their id
  std::shared_ptr<BoundaryConditions> m_bcsArray;

  //! Volumes of voxels where simulation time averaged values should be read
  std::shared_ptr<VoxelCuboidArray> m_sensorArray;

  //! Hashmap with key boundary condition id, value geometry name combined with
  //! this id
  std::vector<std::unordered_set<std::string, std::hash<std::string>>>
      m_idToNameMap;

  //! Hashmap with key geometry name, value all voxel quads sharing this name
  std::unordered_map<std::string,
                     std::unordered_set<VoxelQuad, std::hash<VoxelQuad>>>
      m_nameToQuadMap;

  //! Hashmap with key geo name, value all numerical bc ids sharing this name
  std::unordered_map<std::string, std::unordered_set<int>> m_nameToIdMap;

  /**
   * @brief Stores a pair of boundary condition struct and geometry name string
   * in a hashmap with key as their combined hash, and value the boundary
   * condition. When a name and struct combo is not found in the map, a new
   * boundary condition is stored and id counter incremented.
   *
   * @param bc
   * @param geoName
   */
  void storeType(BoundaryCondition* bc, const std::string& geoName);

  //! Set an element in the array to a voxel id
  inline void set(unsigned int x,
                  unsigned int y,
                  unsigned int z,
                  voxel_t value) {
    (*m_voxelArray)(x - 1, y - 1, z - 1) = value;
  }

  //! Set an element in the array to a voxel id
  inline void set(Vector3<int> v, voxel_t value) {
    set(v.x(), v.y(), v.z(), value);
  }

 public:
  void set(Vector3<int> p,
           BoundaryCondition bc,
           NodeMode::Enum mode,
           std::string name);

  std::unordered_map<Vector3<int>, std::string> getLabels();

  inline std::shared_ptr<VoxelArray> getVoxelArray() { return m_voxelArray; }

  inline std::shared_ptr<BoundaryConditions> getBoundaryConditions() {
    return m_bcsArray;
  }

  inline std::unordered_set<std::string> getObjectNamesById(voxel_t id) {
    if (id > m_idToNameMap.size())
      throw std::invalid_argument(ErrorFormat() << "Invalid key " << id);
    return m_idToNameMap.at(id);
  }

  inline std::unordered_set<VoxelQuad> getQuadsByName(std::string name) {
    if (m_nameToQuadMap.find(name) == m_nameToQuadMap.end())
      throw std::invalid_argument(ErrorFormat() << "Invalid key " << name);
    return m_nameToQuadMap.at(name);
  }

  inline std::vector<int> getIdsByName(std::string name) {
    if (m_nameToIdMap.find(name) == m_nameToIdMap.end())
      throw std::invalid_argument(ErrorFormat() << "Invalid key" << name);
    std::unordered_set<int> idSet = m_nameToIdMap.at(name);
    std::vector<int> ids(idSet.begin(), idSet.end());
    return ids;
  }

  std::unordered_set<voxel_t> getVoxelsByName(std::string name);

  std::vector<std::string> getGeometryNames();

  inline std::shared_ptr<VoxelCuboidArray> getSensors() {
    return m_sensorArray;
  }

  inline unsigned int getNumTypes() { return m_voxelTypeCounter; }

  inline voxel_t get(unsigned int x, unsigned int y, unsigned int z) {
    return (*m_voxelArray)(x - 1, y - 1, z - 1);
  }

  voxel_t inline get(Vector3<int> v) { return get(v.x(), v.y(), v.z()); }

  inline size_t getSizeX() { return m_voxelArray->getSizeX(); }

  inline size_t getSizeY() { return m_voxelArray->getSizeY(); }

  inline size_t getSizeZ() { return m_voxelArray->getSizeZ(); }

  inline size_t getSize() { return getSizeX() * getSizeY() * getSizeZ(); }

  ~VoxelGeometry() {}

  VoxelGeometry();

  VoxelGeometry(size_t nx, size_t ny, size_t nz);
};

std::ostream& operator<<(std::ostream& Str, VoxelGeometry& v);
