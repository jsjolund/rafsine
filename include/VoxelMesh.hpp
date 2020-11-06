#pragma once

#include <osg/Array>
#include <osg/CopyOp>
#include <osg/Geode>
#include <osg/Geometry>
#include <osg/Image>
#include <osg/Material>
#include <osg/Object>
#include <osg/PolygonMode>
#include <osg/PositionAttitudeTransform>
#include <osg/Texture2D>
#include <osg/TextureRectangle>
#include <osg/Vec3>
#include <osgDB/ReadFile>
#include <osgDB/WriteFile>

#include <omp.h>

#include <algorithm>
#include <memory>
#include <string>
#include <utility>

#include "ColorSet.hpp"
#include "DdQq.hpp"
#include "MeshArray.hpp"
#include "VoxelArray.hpp"

/**
 * @brief This class can build and display a mesh based on an voxel array and a
 * color set. It starts by creating quads for each voxel in the array then
 * reducing them by joining of adjacent quads along the same plane.
 */
class VoxelMesh : public osg::Geode {
 protected:
  //! Directions for quad reduction algorithm
  enum Direction { SOUTH = 0, NORTH, EAST, WEST, TOP, BOTTOM };
  //! Mesh geometry
  osg::ref_ptr<osg::Geometry> m_geo;
  // Size of the mesh in voxels
  osg::Vec3i m_size;
  //! Color set used for this mesh
  ColorSet m_colorSet;

  //! Array for complete (reduced) mesh
  MeshArray* m_arrayOrig;
  //! Temporary array for cropped mesh
  MeshArray* m_arrayTmp1;
  //! Temporary array for cropped mesh
  MeshArray* m_arrayTmp2;

  //! How to render the polygons
  osg::PolygonMode::Mode m_polyMode;

  ~VoxelMesh() {
    std::cout << "Destroying voxel mesh" << std::endl;
    delete m_arrayOrig;
    delete m_arrayTmp1;
    delete m_arrayTmp2;
  }

  /**
   * @brief Binds a mesh array to be displayed by openscenegraph
   *
   * @param array
   */
  void bind(MeshArray* array);

  /**
   * @brief Limit a quad polygon to min max. If all vertices are inside range,
   * true is returned. If all vertices are outside, false is returned. If
   * polygon intersects the range, vertices are set to be inside range and true
   * is returned.
   *
   * @param v1 Polygon vertex
   * @param v2 Polygon vertex
   * @param v3 Polygon vertex
   * @param v4 Polygon vertex
   * @param min Minimum value
   * @param max Maximum value
   * @return true
   * @return false
   */
  bool limitPolygon(osg::Vec3* v1,
                    osg::Vec3* v2,
                    osg::Vec3* v3,
                    osg::Vec3* v4,
                    osg::Vec3i min,
                    osg::Vec3i max);

  /**
   * @brief Crops the mesh by min max
   *
   * @param src
   * @param dst
   * @param voxMin
   * @param voxMax
   */
  void crop(MeshArray* src,
            MeshArray* dst,
            osg::Vec3i voxMin,
            osg::Vec3i voxMax);

  /**
   * @brief Construct the 3D mesh using voxel meshing algorithm from
   * https://github.com/mikolalysenko/mikolalysenko.github.com/blob/master/MinecraftMeshes2/js/greedy.js
   *
   * @param voxels
   * @param array Array to put the mesh in
   */
  void buildMeshReduced(std::shared_ptr<VoxelArray> voxels, MeshArray* array);

  /**
   * @brief Construct a part of the voxel mesh
   *
   * @param voxels
   * @param array Array to put the mesh in
   * @param min The minimum coordinate
   * @param max The maximum coordinate
   */
  void buildMeshReduced(std::shared_ptr<VoxelArray> voxels,
                        MeshArray* array,
                        int min[3],
                        int max[3]);

 public:
  /**
   * @brief Constructor with an existing voxel array
   *
   * @param voxels
   */
  explicit VoxelMesh(std::shared_ptr<VoxelArray> voxels);

  /**
   * @brief Load voxel mesh from node file
   *
   * @param filePath
   * @param size Number of lattice sites
   */
  explicit VoxelMesh(const std::string filePath, osg::Vec3i size);

  /**
   * @brief Copy constructor
   *
   * @param voxels
   */
  explicit VoxelMesh(const VoxelMesh& other);

  /**
   * @brief Get the number of lattice sites along the X-axis
   *
   * @return int
   */
  inline int getSizeX() { return m_size.x(); }

  /**
   * @brief Get the number of lattice sites along the Y-axis
   *
   * @return int
   */
  inline int getSizeY() { return m_size.y(); }

  /**
   * @brief Get the number of lattice sites along the Z-axis
   *
   * @return int
   */
  inline int getSizeZ() { return m_size.z(); }

  /**
   * @brief Get the number of lattice sites
   *
   * @return int
   */
  inline int getSize() { return getSizeX() * getSizeY() * getSizeZ(); }

  /**
   * @brief Set how to draw the polygons of the mesh
   *
   * @param mode
   */
  void setPolygonMode(osg::PolygonMode::Mode mode);

  /**
   * @brief Crop the mesh by min max
   *
   * @param voxMin
   * @param voxMax
   */
  void crop(osg::Vec3i voxMin, osg::Vec3i voxMax);

  /**
   * @brief Constructs the voxel mesh
   *
   * @param voxels
   */
  void build(std::shared_ptr<VoxelArray> voxels);

  /**
   * @brief Save mesh to osgb file format
   *
   * @param filePath
   */
  inline void write(std::string filePath) {
    osgDB::writeNodeFile(*this, filePath,
                         new osgDB::Options("Compressor=zlib"));
  }
};
