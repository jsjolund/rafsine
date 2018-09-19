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
class VoxelGeometryObject
{
protected:
  virtual void f(){};

public:
  std::string m_name;
  explicit VoxelGeometryObject(std::string name) : m_name(name){};
};

class VoxelGeometryGroup : public VoxelGeometryObject
{
public:
  std::vector<VoxelGeometryObject *> *m_objs;
  explicit VoxelGeometryGroup(std::string name) : VoxelGeometryObject(name)
  {
    m_objs = new std::vector<VoxelGeometryObject *>();
  }
};

// A plane of voxels
class VoxelGeometryQuad : public VoxelGeometryObject
{
public:
  // World coordinates origin (in m)
  vec3<real> m_origin;
  // Extents (in m)
  vec3<real> m_dir1;
  vec3<real> m_dir2;
  // Mode (fill, overwrite etc.)
  NodeMode::Enum m_mode;
  // Common boundary condition for voxels in this quad
  BoundaryCondition m_bc;

  VoxelGeometryQuad()
      : VoxelGeometryObject(std::string()), m_origin(0, 0, 0), m_dir1(0, 0, 0),
        m_dir2(0, 0, 0), m_mode(NodeMode::Enum::FILL), m_bc(BoundaryCondition()) {}

  VoxelGeometryQuad(std::string name,
                    NodeMode::Enum mode,
                    vec3<real> origin,
                    vec3<real> dir1,
                    vec3<real> dir2,
                    vec3<int> normal,
                    VoxelType::Enum type = VoxelType::Enum::WALL,
                    real temperature = NaN,
                    vec3<real> velocity = vec3<real>(NaN, NaN, NaN),
                    vec3<int> rel_pos = vec3<int>(0, 0, 0))
      : VoxelGeometryObject(name),
        m_bc(BoundaryCondition(-1, type, temperature, velocity, normal, rel_pos)),
        m_origin(origin),
        m_dir1(dir1),
        m_dir2(dir2),
        m_mode(mode) {}
};

namespace std
{
template <>
struct hash<VoxelGeometryQuad>
{
  std::size_t operator()(const VoxelGeometryQuad &quad) const
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

bool operator==(VoxelGeometryQuad const &a, VoxelGeometryQuad const &b);

// A box of voxels
class VoxelGeometryBox : public VoxelGeometryObject
{
public:
  // World coordinates min/max (in m)
  vec3<real> m_min;
  vec3<real> m_max;
  // NaN for no temperature
  real m_temperature;
  // The six quads representing the sides of the box
  std::vector<VoxelGeometryQuad *> m_quads;

  VoxelGeometryBox(std::string name, vec3<real> min, vec3<real> max, real temperature)
      : VoxelGeometryObject(name), m_min(min), m_max(max), m_temperature(temperature)
  {
  }
  VoxelGeometryBox(std::string name, vec3<real> min, vec3<real> max)
      : VoxelGeometryObject(name), m_min(min), m_max(max), m_temperature(NaN)
  {
  }
};

// Class to generate the array of voxels from quads and boxes
class VoxelGeometry
{
private:
  int m_nx, m_ny, m_nz;
  int m_newtype = 1;
  std::shared_ptr<UnitConverter> m_uc;

  std::unordered_map<size_t, BoundaryCondition> m_types;

  // function to get the type from the description
  bool getType(BoundaryCondition *bc, int &id, bool unique);

  // generate a new type of voxel
  // double link voxel type and description
  int createNewVoxelType(BoundaryCondition *bc, bool unique);

  // return the correct voxel type for the boundary
  // create a new one if the boundary does not exist already
  int getBCVoxelType(BoundaryCondition *bc, bool unique);

  // function to compute a new type for intersection of two types
  // or use one already existing
  int getBCIntersectType(vec3<int> position, BoundaryCondition *bc, bool unique);

  // General function to add boundary conditions on a quad
  void addQuadBCNodeUnits(vec3<int> origin, vec3<int> dir1, vec3<int> dir2, VoxelGeometryQuad *geo, bool unique);

  // Set a position in the voxel array to a voxel id
  inline void set(unsigned int x, unsigned int y, unsigned int z, voxel value)
  {
    (*m_data)(x - 1, y - 1, z - 1) = value;
  }
  inline void set(vec3<int> v, voxel value) { set(v.x, v.y, v.z, value); }

public:
  BoundaryConditionsArray m_voxdetail;
  std::unordered_map<voxel, std::unordered_set<VoxelGeometryQuad, std::hash<VoxelGeometryQuad>>> m_quads;
  VoxelArray *m_data;

  inline int getNumTypes() { return m_newtype; }

  inline voxel get(unsigned int x, unsigned int y, unsigned int z)
  {
    return (*m_data)(x - 1, y - 1, z - 1);
  }
  voxel inline get(vec3<int> v) { return get(v.x, v.y, v.z); }

  inline int getNx() { return m_nx; }
  inline int getNy() { return m_ny; }
  inline int getNz() { return m_nz; }

  void saveToFile(std::string filename);
  void loadFromFile(std::string filename);

  // Function to add boundary on a quad. The quad is defined in real units.
  void addQuadBC(VoxelGeometryQuad *geo, bool unique);

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
      real rel_pos,
      bool unique);

  // Function to add a solid box in the domain
  void addSolidBox(VoxelGeometryBox *box, bool unique);

  void createAddSolidBox(
      std::string name,
      real minX, real minY, real minZ,
      real maxX, real maxY, real maxZ,
      real temperature, bool unique);

  // Function to remove the inside of a box
  void makeHollow(vec3<real> min, vec3<real> max,
                  bool xmin, bool ymin, bool zmin,
                  bool xmax, bool ymax, bool zmax);

  void makeHollow(real minX, real minY, real minZ,
                  real maxX, real maxY, real maxZ,
                  bool minXface, bool minYface, bool minZface,
                  bool maxXface, bool maxYface, bool maxZface);

  // Add walls on the domain boundaries
  VoxelGeometryQuad addWallXmin();
  VoxelGeometryQuad addWallXmax();
  VoxelGeometryQuad addWallYmin();
  VoxelGeometryQuad addWallYmax();
  VoxelGeometryQuad addWallZmin();
  VoxelGeometryQuad addWallZmax();

  ~VoxelGeometry() { delete m_data; }

  VoxelGeometry();
  VoxelGeometry(const int nx, const int ny, const int nz, std::shared_ptr<UnitConverter> uc);
};

std::ostream &operator<<(std::ostream &Str, VoxelGeometry &v);