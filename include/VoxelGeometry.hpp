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

#include <glm/vec3.hpp>

#include "BoundaryCondition.hpp"
#include "ErrorFormat.hpp"
#include "UnitConverter.hpp"
#include "Vec3.hpp"
#include "VoxelArray.hpp"
#include "VoxelObject.hpp"

#define DEFAULT_GEOMETRY_NAME "geometry"

// Class to generate the array of voxels from quads and boxes
class VoxelGeometry {
 protected:
  const int m_nx, m_ny, m_nz;
  int m_newtype;
  std::shared_ptr<VoxelArray> m_voxelArray;
  std::shared_ptr<BoundaryConditions> m_bcsArray;
  std::shared_ptr<VoxelVolumeArray> m_sensorArray;
  std::shared_ptr<UnitConverter> m_uc;
  std::unordered_map<size_t, BoundaryCondition> m_types;
  std::unordered_map<voxel,
                     std::unordered_set<std::string, std::hash<std::string>>>
      m_voxNameMap;
  std::unordered_map<std::string,
                     std::unordered_set<VoxelQuad, std::hash<VoxelQuad>>>
      m_nameQuadMap;

  voxel storeType(BoundaryCondition *bc, const std::string &quadName);

  // Set a position in the voxel array to a voxel id
  inline void set(unsigned int x, unsigned int y, unsigned int z, voxel value) {
    (*m_voxelArray)(x - 1, y - 1, z - 1) = value;
  }

  inline void set(vec3<int> v, voxel value) { set(v.x, v.y, v.z, value); }

 public:
  std::unordered_map<glm::ivec3, std::string> getLabels();

  inline std::shared_ptr<VoxelArray> getVoxelArray() { return m_voxelArray; }

  inline std::shared_ptr<BoundaryConditions> getBoundaryConditions() {
    return m_bcsArray;
  }

  inline std::unordered_set<std::string> getObjectNamesById(voxel id) {
    return m_voxNameMap.at(id);
  }

  inline std::unordered_set<VoxelQuad> getQuadsByName(std::string name) {
    return m_nameQuadMap.at(name);
  }

  std::unordered_set<voxel> getVoxelsByName(std::string name);

  std::vector<std::string> getGeometryNames();

  inline std::shared_ptr<VoxelVolumeArray> getSensors() {
    return m_sensorArray;
  }

  inline int getNumTypes() { return m_newtype; }

  inline voxel get(unsigned int x, unsigned int y, unsigned int z) {
    return (*m_voxelArray)(x - 1, y - 1, z - 1);
  }

  voxel inline get(vec3<int> v) { return get(v.x, v.y, v.z); }

  inline int getNx() { return m_nx; }

  inline int getNy() { return m_ny; }

  inline int getNz() { return m_nz; }

  inline int getSize() { return m_nx * m_ny * m_nz; }

  ~VoxelGeometry() {}

  VoxelGeometry();

  VoxelGeometry(const int nx, const int ny, const int nz,
                std::shared_ptr<UnitConverter> uc);
};

std::ostream &operator<<(std::ostream &Str, VoxelGeometry &v);
