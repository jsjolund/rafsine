#pragma once

#include <boost/algorithm/string.hpp>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <string>
#include <iostream>
#include <sstream>
#include <iomanip>
#include <fstream>

#include "UnitConverter.hpp"
#include "BoundaryCondition.hpp"
#include "Primitives.hpp"
#include "Voxel.hpp"
#include "ErrorFormat.hpp"

#define DEFAULT_GEOMETRY_NAME "geometry"

namespace NodeMode
{
enum Enum
{
  OVERWRITE,
  INTERSECT,
  FILL
};
}

std::ostream &operator<<(std::ostream &os, NodeMode::Enum v);

// A base class only holding a name string
class VoxelObject
{
public:
  std::string m_name;
  explicit VoxelObject(std::string name) : m_name(name){};
};

// A plane of voxels
class VoxelQuad : public VoxelObject
{
public:
  // World coordinates origin (in m)
  vec3<real> m_origin;
  // Extents (in m)
  vec3<real> m_dir1;
  vec3<real> m_dir2;
  // Discretized coordinates and extents in lattice units
  vec3<int> m_voxOrigin;
  vec3<int> m_voxDir1;
  vec3<int> m_voxDir2;

  std::unordered_set<BoundaryCondition, std::hash<BoundaryCondition>> m_intersectingBcs;

  // Mode (fill, overwrite etc.)
  NodeMode::Enum m_mode;
  // Common boundary condition for voxels in this quad
  BoundaryCondition m_bc;

  VoxelQuad()
      : VoxelObject(std::string()),
        m_origin(0, 0, 0),
        m_dir1(0, 0, 0),
        m_dir2(0, 0, 0),
        m_mode(NodeMode::Enum::FILL),
        m_bc(BoundaryCondition()),
        m_voxOrigin(0, 0, 0),
        m_voxDir1(0, 0, 0),
        m_voxDir2(0, 0, 0) {}

  VoxelQuad(std::string name,
            NodeMode::Enum mode,
            vec3<real> origin,
            vec3<real> dir1,
            vec3<real> dir2,
            vec3<int> normal,
            VoxelType::Enum type = VoxelType::Enum::WALL,
            real temperature = NaN,
            vec3<real> velocity = vec3<real>(NaN, NaN, NaN),
            vec3<int> rel_pos = vec3<int>(0, 0, 0))
      : VoxelObject(name),
        m_bc(BoundaryCondition(-1, type, temperature, velocity, normal, rel_pos)),
        m_origin(origin),
        m_dir1(dir1),
        m_dir2(dir2),
        m_mode(mode),
        m_voxOrigin(0, 0, 0),
        m_voxDir1(0, 0, 0),
        m_voxDir2(0, 0, 0) {}

  VoxelQuad(std::string name,
            NodeMode::Enum mode,
            vec3<int> origin,
            vec3<int> dir1,
            vec3<int> dir2,
            vec3<int> normal,
            VoxelType::Enum type = VoxelType::Enum::WALL,
            real temperature = NaN,
            vec3<real> velocity = vec3<real>(NaN, NaN, NaN),
            vec3<int> rel_pos = vec3<int>(0, 0, 0))
      : VoxelObject(name),
        m_bc(BoundaryCondition(-1, type, temperature, velocity, normal, rel_pos)),
        m_origin(origin),
        m_dir1(dir1),
        m_dir2(dir2),
        m_mode(mode),
        m_voxOrigin(origin),
        m_voxDir1(dir1),
        m_voxDir2(dir2) {}
};

namespace std
{
template <>
struct hash<VoxelQuad>
{
  std::size_t operator()(const VoxelQuad &quad) const
  {
    using std::hash;
    using std::size_t;
    size_t seed = 0;
    ::hash_combine(seed, quad.m_origin.x);
    ::hash_combine(seed, quad.m_origin.y);
    ::hash_combine(seed, quad.m_origin.z);
    ::hash_combine(seed, quad.m_dir1.x);
    ::hash_combine(seed, quad.m_dir1.y);
    ::hash_combine(seed, quad.m_dir1.z);
    ::hash_combine(seed, quad.m_dir2.x);
    ::hash_combine(seed, quad.m_dir2.y);
    ::hash_combine(seed, quad.m_dir2.z);
    ::hash_combine(seed, quad.m_mode);
    ::hash_combine(seed, quad.m_name);
    return seed;
  }
};
} // namespace std

bool operator==(VoxelQuad const &a, VoxelQuad const &b);

// A box of voxels
class VoxelBox : public VoxelObject
{
public:
  // World coordinates min/max (in m)
  vec3<real> m_min;
  vec3<real> m_max;
  // NaN for no temperature
  real m_temperature;
  // The six quads representing the sides of the box
  std::vector<VoxelQuad *> m_quads;

  VoxelBox(std::string name, vec3<real> min, vec3<real> max, real temperature)
      : VoxelObject(name), m_min(min), m_max(max), m_temperature(temperature)
  {
  }
  VoxelBox(std::string name, vec3<real> min, vec3<real> max)
      : VoxelObject(name), m_min(min), m_max(max), m_temperature(NaN)
  {
  }
};

// Class to generate the array of voxels from quads and boxes
class VoxelGeometry
{
private:
  int m_nx, m_ny, m_nz;
  int m_newtype;
  std::shared_ptr<UnitConverter> m_uc;

  voxel storeType(BoundaryCondition &bc, std::string &quadName);

  // General function to add boundary conditions on a quad
  void addQuadBCNodeUnits(VoxelQuad *geo);

  // Set a position in the voxel array to a voxel id
  inline void set(unsigned int x, unsigned int y, unsigned int z, voxel value)
  {
    (*m_voxelArray)(x - 1, y - 1, z - 1) = value;
  }
  inline void set(vec3<int> v, voxel value) { set(v.x, v.y, v.z, value); }

  std::unordered_map<size_t, BoundaryCondition> m_types;
  std::unordered_map<voxel, std::unordered_set<std::string, std::hash<std::string>>> m_voxNameMap;
  std::unordered_map<std::string, std::unordered_set<VoxelQuad, std::hash<VoxelQuad>>> m_nameQuadMap;

  BoundaryConditionsArray m_bcsArray;
  VoxelArray *m_voxelArray;

public:
  inline VoxelArray *getVoxelArray() { return m_voxelArray; }
  inline BoundaryConditionsArray *getBoundaryConditions() { return &m_bcsArray; }
  inline std::unordered_set<std::string> getObjectNamesById(voxel id) { return m_voxNameMap.at(id); }
  inline std::unordered_set<VoxelQuad> getQuadsByName(std::string name) { return m_nameQuadMap.at(name); }
  inline std::unordered_set<voxel> getVoxelsByName(std::string name)
  {
    std::unordered_set<VoxelQuad> quads = m_nameQuadMap.at(name);
    std::unordered_set<voxel> voxIds;
    for (const VoxelQuad quad : quads)
    {
      voxIds.insert(quad.m_bc.m_id);
      for (BoundaryCondition bc : quad.m_intersectingBcs)
        voxIds.insert(bc.m_id);
    }
    return voxIds;
  }

  inline int getNumTypes() { return m_newtype; }

  inline voxel get(unsigned int x, unsigned int y, unsigned int z)
  {
    return (*m_voxelArray)(x - 1, y - 1, z - 1);
  }
  voxel inline get(vec3<int> v) { return get(v.x, v.y, v.z); }

  inline int getNx() { return m_nx; }
  inline int getNy() { return m_ny; }
  inline int getNz() { return m_nz; }

  void saveToFile(std::string filename);
  void loadFromFile(std::string filename);

  // Function to add boundary on a quad. The quad is defined in real units.
  void addQuadBC(VoxelQuad *geo);

  void createAddQuadBC(
      std::string name,
      std::string mode,
      real originX, real originY, real originZ,
      real dir1X, real dir1Y, real dir1Z,
      real dir2X, real dir2Y, real dir2Z,
      int normalX, int normalY, int normalZ,
      std::string typeBC,
      std::string temperatureType,
      real temperature,
      real velocityX, real velocityY, real velocityZ,
      real rel_pos);

  // Function to add a solid box in the domain
  void addSolidBox(VoxelBox *box);

  void createAddSolidBox(
      std::string name,
      real minX, real minY, real minZ,
      real maxX, real maxY, real maxZ,
      real temperature);

  // Function to remove the inside of a box
  void makeHollow(vec3<real> min, vec3<real> max,
                  bool xmin, bool ymin, bool zmin,
                  bool xmax, bool ymax, bool zmax);

  void makeHollow(real minX, real minY, real minZ,
                  real maxX, real maxY, real maxZ,
                  bool minXface, bool minYface, bool minZface,
                  bool maxXface, bool maxYface, bool maxZface);

  // Add walls on the domain boundaries
  VoxelQuad addWallXmin();
  VoxelQuad addWallXmax();
  VoxelQuad addWallYmin();
  VoxelQuad addWallYmax();
  VoxelQuad addWallZmin();
  VoxelQuad addWallZmax();

  ~VoxelGeometry() { delete m_voxelArray; }

  VoxelGeometry();
  VoxelGeometry(const int nx, const int ny, const int nz, std::shared_ptr<UnitConverter> uc);
};

std::ostream &operator<<(std::ostream &Str, VoxelGeometry &v);