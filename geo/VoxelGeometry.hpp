#pragma once

#include <glm/glm.hpp>
#include <string>
#include <unordered_map>
#include <vector>
#include "../sim/BoundaryCondition.hpp"
#include "UnitConverter.hpp"
#include <iostream>
#include <sstream>
#include <iomanip>

using glm::ivec3;
using glm::vec3;
using std::string;

template <typename T>
int sgn(T val)
{
  return (T(0) < val) - (val < T(0));
}

class VoxelGeometry
{
private:
  int nx, ny, nz;
  int ***data;
  int newtype = 1;
  UnitConverter *uc;
  std::unordered_map<size_t, BoundaryCondition> types;

public:
  std::vector<BoundaryCondition> voxdetail;
  void inline set(int x, int y, int z, int value) { (data)[x-1][y-1][z-1] = value; }
  int inline get(int x, int y, int z) { return (data)[x-1][y-1][z-1]; }
  int inline get(ivec3 v) { return get(v.x, v.y, v.z); }

  void saveToFile(string filename);

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
  int getBCIntersectType(ivec3 position, BoundaryCondition *bc);

  // General function to add boundary conditions on a quad
  int addQuadBCNodeUnits(ivec3 origin, ivec3 dir1, ivec3 dir2, DomainGeometryQuad *geo);

  // function to add boundary on a quad
  // the quad is defined in real units
  int addQuadBC(DomainGeometryQuad *geo);

  // Add walls on the domain boundaries
  void addWallXmin();
  // Add walls on the domain boundaries
  void addWallXmax();
  // Add walls on the domain boundaries
  void addWallYmin();
  // Add walls on the domain boundaries
  void addWallYmax();
  // Add walls on the domain boundaries
  void addWallZmin();
  // Add walls on the domain boundaries
  void addWallZmax();

  void addSolidBox(DomainGeometryBox *box);

  ~VoxelGeometry() { delete data; }

  VoxelGeometry(const int nx, const int ny, const int nz, UnitConverter *uc);
};