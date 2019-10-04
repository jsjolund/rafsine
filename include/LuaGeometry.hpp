#pragma once

#include <memory>
#include <string>

#include "VoxelGeometry.hpp"

class LuaGeometry : public VoxelGeometry {
 private:
  //! Convert between meters and lattice units
  std::shared_ptr<UnitConverter> m_uc;

 public:
  LuaGeometry(const int nx, const int ny, const int nz,
              std::shared_ptr<UnitConverter> uc)
      : VoxelGeometry(nx, ny, nz), m_uc(uc) {}

  // General function to add boundary conditions on a quad
  void addQuadBCNodeUnits(VoxelQuad *geo);

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
  void makeHollow(glm::vec3 min, glm::vec3 max, bool xmin, bool ymin, bool zmin,
                  bool xmax, bool ymax, bool zmax);

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
};
