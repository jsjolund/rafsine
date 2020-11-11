#pragma once

#include <memory>
#include <string>

#include "Vector3.hpp"
#include "VoxelGeometry.hpp"

/**
 * @brief Constructs voxel geometry from LUA script
 */
class LuaGeometry : public VoxelGeometry {
 private:
  //! Convert between meters and lattice units
  std::shared_ptr<UnitConverter> m_uc;

  /**
   * @brief General function to add boundary conditions on a quad
   *
   * @param geo
   */
  void addQuadBCNodeUnits(VoxelQuad* geo);

 public:
  /**
   * @brief Construct a new Lua Geometry with given size and unit conversion
   *
   * @param nx
   * @param ny
   * @param nz
   * @param uc
   */
  LuaGeometry(const int nx,
              const int ny,
              const int nz,
              std::shared_ptr<UnitConverter> uc)
      : VoxelGeometry(nx, ny, nz), m_uc(uc) {}

  /**
   * @brief Add boundary on a quad. The quad is defined in real units.
   *
   */
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

  /**
   * @brief Adds a time averaging sensor area to the domain
   *
   * @param name
   * @param minX
   * @param minY
   * @param minZ
   * @param maxX
   * @param maxY
   * @param maxZ
   */
  void addSensor(std::string name,
                 real_t minX,
                 real_t minY,
                 real_t minZ,
                 real_t maxX,
                 real_t maxY,
                 real_t maxZ);

  /**
   * @brief Add a solid box in the domain
   *
   * @param name
   * @param minX
   * @param minY
   * @param minZ
   * @param maxX
   * @param maxY
   * @param maxZ
   * @param temperature
   */
  void addSolidBox(std::string name,
                   real_t minX,
                   real_t minY,
                   real_t minZ,
                   real_t maxX,
                   real_t maxY,
                   real_t maxZ,
                   real_t temperature);

  /**
   * @brief Add a solid sphere in the domain
   *
   * @param name
   * @param originX
   * @param originY
   * @param originZ
   * @param radius
   * @param temperature
   */
  void addSolidSphere(std::string name,
                      real_t originX,
                      real_t originY,
                      real_t originZ,
                      real_t radius,
                      real_t temperature);

  /**
   * @brief Remove the inside of a box
   *
   * @param min
   * @param max
   * @param xmin
   * @param ymin
   * @param zmin
   * @param xmax
   * @param ymax
   * @param zmax
   */
  void makeHollow(Vector3<real_t> min,
                  Vector3<real_t> max,
                  bool xmin,
                  bool ymin,
                  bool zmin,
                  bool xmax,
                  bool ymax,
                  bool zmax);

  /**
   * @brief Remove the inside of a box
   *
   * @param minX
   * @param minY
   * @param minZ
   * @param maxX
   * @param maxY
   * @param maxZ
   * @param minXface
   * @param minYface
   * @param minZface
   * @param maxXface
   * @param maxYface
   * @param maxZface
   */
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

  /**
   * @brief Add walls on the domain boundaries
   */
  void addWallXmin();
  /**
   * @brief Add walls on the domain boundaries
   */
  void addWallXmax();
  /**
   * @brief Add walls on the domain boundaries
   */
  void addWallYmin();
  /**
   * @brief Add walls on the domain boundaries
   */
  void addWallYmax();
  /**
   * @brief Add walls on the domain boundaries
   */
  void addWallZmin();
  /**
   * @brief Add walls on the domain boundaries
   */
  void addWallZmax();
};
