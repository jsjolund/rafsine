#pragma once

#include <memory>
#include <string>

#include "Vector3.hpp"
#include "VoxelGeometry.hpp"

class LuaGeometry : public VoxelGeometry {
 private:
  //! Convert between meters and lattice units
  std::shared_ptr<UnitConverter> m_uc;

  // General function to add boundary conditions on a quad
  void addQuadBCNodeUnits(VoxelQuad* geo);

 public:
  LuaGeometry(const int nx,
              const int ny,
              const int nz,
              std::shared_ptr<UnitConverter> uc)
      : VoxelGeometry(nx, ny, nz), m_uc(uc) {}

  // Function to add boundary on a quad. The quad is defined in real units.
  void addQuadBC(std::string name,
                 std::string mode,
                 real_t originX,
                 real_t originY,
                 real_t originZ,
                 real_t dir1X,
                 real_t dir1Y,
                 real_t dir1Z,
                 real_t dir2X,
                 real_t dir2Y,
                 real_t dir2Z,
                 int normalX,
                 int normalY,
                 int normalZ,
                 std::string typeBC,
                 std::string temperatureType,
                 real_t temperature,
                 real_t velocityX,
                 real_t velocityY,
                 real_t velocityZ,
                 real_t rel_pos,
                 real_t tau1,
                 real_t tau2,
                 real_t lambda);

  void addSensor(std::string name,
                 real_t minX,
                 real_t minY,
                 real_t minZ,
                 real_t maxX,
                 real_t maxY,
                 real_t maxZ);

  // Function to add a solid box in the domain
  void addSolidBox(std::string name,
                   real_t minX,
                   real_t minY,
                   real_t minZ,
                   real_t maxX,
                   real_t maxY,
                   real_t maxZ,
                   real_t temperature);

  // Function to add a solid sphere in the domain
  void addSolidSphere(std::string name,
                      real_t originX,
                      real_t originY,
                      real_t originZ,
                      real_t radius,
                      real_t temperature);

  // Function to remove the inside of a box
  void makeHollow(Vector3<real_t> min,
                  Vector3<real_t> max,
                  bool xmin,
                  bool ymin,
                  bool zmin,
                  bool xmax,
                  bool ymax,
                  bool zmax);

  void makeHollow(real_t minX,
                  real_t minY,
                  real_t minZ,
                  real_t maxX,
                  real_t maxY,
                  real_t maxZ,
                  bool minXface,
                  bool minYface,
                  bool minZface,
                  bool maxXface,
                  bool maxYface,
                  bool maxZface);

  // Add walls on the domain boundaries
  void addWallXmin();
  void addWallXmax();
  void addWallYmin();
  void addWallYmax();
  void addWallZmin();
  void addWallZmax();
};
