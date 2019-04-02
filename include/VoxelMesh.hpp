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

#include <omp.h>

#include <algorithm>
#include <memory>
#include <string>
#include <utility>

#include "ColorSet.hpp"
#include "DdQq.hpp"
#include "MeshArray.hpp"
#include "VoxelArray.hpp"

namespace VoxelMeshType {
enum Enum { FULL, REDUCED };
}

/**
 * @brief This class can build and display a mesh based on an voxel array and a
 * color set
 *
 */
class VoxelMesh : public osg::Geode {
 protected:
  enum Direction { SOUTH = 0, NORTH, EAST, WEST, TOP, BOTTOM };
  // Size of the mesh in voxels
  osg::Vec3i m_size;
  //! Color set used for this mesh
  ColorSet m_colorSet;
  //! Mesh geometry
  osg::ref_ptr<osg::Geometry> m_geo;

  MeshArray *m_arrayOrig;
  MeshArray *m_arrayTmp1;
  MeshArray *m_arrayTmp2;

  //! How to render the polygons
  osg::PolygonMode::Mode m_polyMode;

  ~VoxelMesh() {
    std::cout << "Destroying voxel mesh" << std::endl;
    delete m_arrayOrig;
    delete m_arrayTmp1;
    delete m_arrayTmp2;
  }

  void bind(MeshArray *array);

  bool limitPolygon(osg::Vec3 *v1, osg::Vec3 *v2, osg::Vec3 *v3, osg::Vec3 *v4,
                    osg::Vec3i min, osg::Vec3i max);

  void crop(MeshArray *src, MeshArray *dst, osg::Vec3i voxMin,
            osg::Vec3i voxMax);

  /**
   * @brief Construct the 3D mesh, fill the vertex, normal and color arrays
   *
   * @param array
   */
  void buildMeshFull(MeshArray *array);

  /**
   * @brief Construct the 3D mesh using voxel meshing algorithm from
   * https://github.com/mikolalysenko/mikolalysenko.github.com/blob/master/MinecraftMeshes2/js/greedy.js
   *
   * @param array Array to put the mesh in
   */
  void buildMeshReduced(std::shared_ptr<VoxelArray> voxels, MeshArray *array);
  /**
   * @brief Construct a part of the voxel mesh
   *
   * @param array Array to put the mesh in
   * @param min The minimum coordinate
   * @param max The maximum coordinate
   */
  void buildMeshReduced(std::shared_ptr<VoxelArray> voxels, MeshArray *array,
                        int min[3], int max[3]);

 public:
  /**
   * @brief Constructor with an existing voxel array
   *
   * @param voxels
   */
  explicit VoxelMesh(std::shared_ptr<VoxelArray> voxels);

  /**
   * @brief Copy constructor
   *
   * @param voxels
   */
  explicit VoxelMesh(const VoxelMesh &other);

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

  void crop(osg::Vec3i voxMin, osg::Vec3i voxMax);

  void build(std::shared_ptr<VoxelArray> voxels, VoxelMeshType::Enum type);
};
