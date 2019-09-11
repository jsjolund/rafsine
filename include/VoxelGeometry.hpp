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

#include <boost/algorithm/string.hpp>
#include <glm/vec3.hpp>

#include "BoundaryCondition.hpp"
#include "ErrorFormat.hpp"
#include "Primitives.hpp"
#include "UnitConverter.hpp"
#include "VoxelArray.hpp"
#include "VoxelObject.hpp"

#define DEFAULT_GEOMETRY_NAME "geometry"

// Class to generate the array of voxels from quads and boxes
class VoxelGeometry {
 private:
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

  // General function to add boundary conditions on a quad
  void addQuadBCNodeUnits(VoxelQuad *geo);

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

  void saveToFile(std::string filename);
  void loadFromFile(std::string filename);

  // Function to add boundary on a quad. The quad is defined in real units.
  void addQuadBC(std::string name, std::string mode, real originX, real originY,
                 real originZ, real dir1X, real dir1Y, real dir1Z, real dir2X,
                 real dir2Y, real dir2Z, int normalX, int normalY, int normalZ,
                 std::string typeBC, std::string temperatureType,
                 real temperature, real velocityX, real velocityY,
                 real velocityZ, real rel_pos);

  void addSensor(std::string name, real minX, real minY, real minZ, real maxX,
                 real maxY, real maxZ);

  // Function to add a solid box in the domain
  void addSolidBox(std::string name, real minX, real minY, real minZ, real maxX,
                   real maxY, real maxZ, real temperature);

  // Function to remove the inside of a box
  void makeHollow(vec3<real> min, vec3<real> max, bool xmin, bool ymin,
                  bool zmin, bool xmax, bool ymax, bool zmax);

  void makeHollow(real minX, real minY, real minZ, real maxX, real maxY,
                  real maxZ, bool minXface, bool minYface, bool minZface,
                  bool maxXface, bool maxYface, bool maxZface);

  // Add walls on the domain boundaries
  void addWallXmin();
  void addWallXmax();
  void addWallYmin();
  void addWallYmax();
  void addWallZmin();
  void addWallZmax();

  ~VoxelGeometry() {}

  VoxelGeometry();
  VoxelGeometry(const int nx, const int ny, const int nz,
                std::shared_ptr<UnitConverter> uc);
};

std::ostream &operator<<(std::ostream &Str, VoxelGeometry &v);
