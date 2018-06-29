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

#include "../sim/UnitConverter.hpp"
#include "../sim/BoundaryCondition.hpp"
#include "Primitives.hpp"
#include "Voxel.hpp"

using std::string;

enum NodeMode
{
  OVERWRITE,
  INTERSECT,
  FILL
};

template <typename T>
int sgn(T val)
{
  return (T(0) < val) - (val < T(0));
}

class VoxelGeometry
{
private:
  int nx, ny, nz;
  int newtype = 1;
  // UnitConverter *uc;
  std::unordered_map<size_t, BoundaryCondition> types;

  void initVoxData(int nx, int ny, int nz);

public:
  VoxelArray *data;
  std::vector<BoundaryCondition> voxdetail;

  void inline set(unsigned int x, unsigned int y, unsigned int z, int value)
  {
    (*data)(x - 1, y - 1, z - 1) = value;
  }
  void inline set(vec3<int> v, int value) { set(v.x, v.y, v.z, value); }
  int inline get(unsigned int x, unsigned int y, unsigned int z)
  {
    return (*data)(x - 1, y - 1, z - 1);
  }
  int inline get(vec3<int> v) { return get(v.x, v.y, v.z); }
  int inline getNx() { return nx; }
  int inline getNy() { return ny; }
  int inline getNz() { return nz; }

  void saveToFile(string filename);
  void loadFromFile(string filename);

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
                   std::vector<VoxelGeometryObject> *quads,
                   UnitConverter *uc);

  ~VoxelGeometry() { delete data; }

  VoxelGeometry();
  VoxelGeometry(const int nx, const int ny, const int nz, UnitConverter *uc);
};

std::ostream &operator<<(std::ostream &Str, VoxelGeometry &v);