#pragma once

#include <boost/algorithm/string.hpp>
#include <glm/glm.hpp>
#include <unordered_map>
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
  std::string name;
  explicit VoxelGeometryObject(std::string name) : name(name){};
};

class VoxelGeometryGroup : public VoxelGeometryObject
{
public:
  std::vector<VoxelGeometryObject *> *objs;
  explicit VoxelGeometryGroup(std::string name) : VoxelGeometryObject(name)
  {
    objs = new std::vector<VoxelGeometryObject *>();
  }
};

// A plane of voxels
class VoxelGeometryQuad : public VoxelGeometryObject
{
public:
  // World coordinates origin (in m)
  vec3<real> origin;
  // Extents (in m)
  vec3<real> dir1;
  vec3<real> dir2;
  // Mode (fill, overwrite etc.)
  NodeMode::Enum mode;
  // Common boundary condition for voxels in this quad
  BoundaryCondition bc;

  VoxelGeometryQuad()
      : VoxelGeometryObject(std::string()), origin(0, 0, 0), dir1(0, 0, 0),
        dir2(0, 0, 0), mode(NodeMode::Enum::FILL), bc(BoundaryCondition()) {}

  VoxelGeometryQuad(std::string name,
                    NodeMode::Enum mode,
                    vec3<real> origin,
                    vec3<real> dir1,
                    vec3<real> dir2,
                    vec3<int> normal,
                    VoxelType::Enum type = VoxelType::Enum::WALL,
                    real temperature = NaN,
                    vec3<real> velocity = vec3<real>(0, 0, 0),
                    vec3<int> rel_pos = vec3<int>(0, 0, 0))
      : VoxelGeometryObject(name), origin(origin), dir1(dir1),
        dir2(dir2), mode(mode),
        bc(BoundaryCondition(-1, type, temperature, velocity, normal, rel_pos))
  {
  }
};
// A box of voxels
class VoxelGeometryBox : public VoxelGeometryObject
{
public:
  // World coordinates min/max (in m)
  vec3<real> min;
  vec3<real> max;
  // NaN for no temperature
  real temperature;
  // The six quads representing the sides of the box
  std::vector<VoxelGeometryQuad *> quads;

  VoxelGeometryBox(std::string name, vec3<real> min, vec3<real> max, real temperature)
      : VoxelGeometryObject(name), min(min), max(max), temperature(temperature)
  {
  }
  VoxelGeometryBox(std::string name, vec3<real> min, vec3<real> max)
      : VoxelGeometryObject(name), min(min), max(max), temperature(NaN)
  {
  }
};

// Class to generate the array of voxels from quads and boxes
class VoxelGeometry
{
private:
  int nx, ny, nz;
  int newtype = 1;

  std::unordered_map<size_t, BoundaryCondition> types;
  std::vector<BoundaryCondition> voxdetail;

  // function to get the type from the description
  bool getType(BoundaryCondition *bc, int &id);

  // function to set the type from a description
  void setType(BoundaryCondition *bc, int value);

  // generate a new type of voxel
  // double link voxel type and description
  int createNewVoxelType(BoundaryCondition *bc);

  // return the correct voxel type for the boundary
  // create a new one if the boundary does not exist already
  int getBCVoxelType(BoundaryCondition *bc);

  // function to compute a new type for intersection of two types
  // or use one already existing
  int getBCIntersectType(vec3<int> position, BoundaryCondition *bc);

  // General function to add boundary conditions on a quad
  int addQuadBCNodeUnits(vec3<int> origin, vec3<int> dir1, vec3<int> dir2, VoxelGeometryQuad *geo);

  // Set a position in the voxel array to a voxel id
  void inline set(unsigned int x, unsigned int y, unsigned int z, voxel value)
  {
    (*data)(x - 1, y - 1, z - 1) = value;
  }
  void inline set(vec3<int> v, voxel value) { set(v.x, v.y, v.z, value); }

public:
  VoxelArray *data;

  voxel inline get(unsigned int x, unsigned int y, unsigned int z)
  {
    return (*data)(x - 1, y - 1, z - 1);
  }
  voxel inline get(vec3<int> v) { return get(v.x, v.y, v.z); }

  int inline getNx() { return nx; }
  int inline getNy() { return ny; }
  int inline getNz() { return nz; }

  void saveToFile(std::string filename);
  void loadFromFile(std::string filename);

  // function to add boundary on a quad
  // the quad is defined in real units
  int addQuadBC(VoxelGeometryQuad *geo, UnitConverter *uc);

  // Add walls on the domain boundaries
  VoxelGeometryQuad addWallXmin();
  // Add walls on the domain boundaries
  VoxelGeometryQuad addWallXmax();
  // Add walls on the domain boundaries
  VoxelGeometryQuad addWallYmin();
  // Add walls on the domain boundaries
  VoxelGeometryQuad addWallYmax();
  // Add walls on the domain boundaries
  VoxelGeometryQuad addWallZmin();
  // Add walls on the domain boundaries
  VoxelGeometryQuad addWallZmax();

  // function to remove the inside of a box
  void makeHollow(vec3<real> min, vec3<real> max,
                  bool xmin, bool ymin, bool zmin,
                  bool xmax, bool ymax, bool zmax, UnitConverter *uc);

  // function to add a solid box in the domain
  void addSolidBox(VoxelGeometryBox *box,
                   UnitConverter *uc);

  ~VoxelGeometry() { delete data; }

  VoxelGeometry();
  VoxelGeometry(const int nx, const int ny, const int nz, UnitConverter *uc);
};

std::ostream &operator<<(std::ostream &Str, VoxelGeometry &v);