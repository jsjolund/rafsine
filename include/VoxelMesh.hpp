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

#include <algorithm>
#include <string>

#include "ColorSet.hpp"
#include "Voxel.hpp"

namespace VoxelMeshType {
enum Enum { FULL, REDUCED };
}

/**
 * @brief  This class can build and display a mesh based on an voxel array and a
 * color set
 *
 */
class VoxelMesh : public osg::Geometry {
 protected:
  enum Direction { SOUTH = 0, NORTH, EAST, WEST, TOP, BOTTOM };

  //! Voxels to base the mesh on
  VoxelArray *m_voxels;
  //! Color set used for this mesh
  ColorSet *m_colorSet;
  //! World transform
  osg::ref_ptr<osg::PositionAttitudeTransform> m_transform;

  //! Vertices from the generated mesh
  osg::ref_ptr<osg::Vec3Array> m_vertexArray;
  //! Color of each vertex
  osg::ref_ptr<osg::Vec4Array> m_colorArray;
  //! Plane normals
  osg::ref_ptr<osg::Vec3Array> m_normalsArray;
  //! Texture coordinates
  osg::ref_ptr<osg::Vec2Array> m_texCoordArray;

  osg::ref_ptr<osg::Vec3Array> m_vertexArrayTmp1;
  osg::ref_ptr<osg::Vec4Array> m_colorArrayTmp1;
  osg::ref_ptr<osg::Vec3Array> m_normalsArrayTmp1;
  osg::ref_ptr<osg::Vec2Array> m_texCoordArrayTmp1;

  osg::ref_ptr<osg::Vec3Array> m_vertexArrayTmp2;
  osg::ref_ptr<osg::Vec4Array> m_colorArrayTmp2;
  osg::ref_ptr<osg::Vec3Array> m_normalsArrayTmp2;
  osg::ref_ptr<osg::Vec2Array> m_texCoordArrayTmp2;

  //! How to render the polygons
  osg::PolygonMode::Mode m_polyMode;

  ~VoxelMesh() { delete m_colorSet; }

  void bind(osg::ref_ptr<osg::Vec3Array> vertexArray,
            osg::ref_ptr<osg::Vec4Array> colorArray,
            osg::ref_ptr<osg::Vec3Array> normalArray,
            osg::ref_ptr<osg::Vec2Array> texCoordArray);

  bool limitPolygon(osg::Vec3 *v1, osg::Vec3 *v2, osg::Vec3 *v3, osg::Vec3 *v4,
                    osg::Vec3i min, osg::Vec3i max);
  void swap();

  void clear(osg::ref_ptr<osg::Vec3Array> vertexArray,
             osg::ref_ptr<osg::Vec4Array> colorArray,
             osg::ref_ptr<osg::Vec3Array> normalArray,
             osg::ref_ptr<osg::Vec2Array> texCoordArray);

  void crop(osg::ref_ptr<osg::Vec3Array> srcVertices,
            osg::ref_ptr<osg::Vec4Array> srcColors,
            osg::ref_ptr<osg::Vec3Array> srcNormals,
            osg::ref_ptr<osg::Vec2Array> srcTexCoords,
            osg::ref_ptr<osg::Vec3Array> dstVertices,
            osg::ref_ptr<osg::Vec4Array> dstColors,
            osg::ref_ptr<osg::Vec3Array> dstNormals,
            osg::ref_ptr<osg::Vec2Array> dstTexCoords, osg::Vec3i voxMin,
            osg::Vec3i voxMax);

  /**
   * @brief Construct the 3D mesh, fill the vertex, normal and color arrays
   *
   * @param voxMin
   * @param voxMax
   */
  void buildMeshFull(osg::ref_ptr<osg::Vec3Array> vertices,
                     osg::ref_ptr<osg::Vec4Array> colors,
                     osg::ref_ptr<osg::Vec3Array> normals,
                     osg::ref_ptr<osg::Vec2Array> texCoords);

  /**
   * @brief Construct the 3D mesh, fill the vertex, normal and color arrays.
   * Vertex reduced version from
   * https://github.com/mikolalysenko/mikolalysenko.github.com/blob/master/MinecraftMeshes2/js/greedy.js
   *
   * @param voxMin
   * @param voxMax
   */
  void buildMeshReduced(osg::ref_ptr<osg::Vec3Array> vertices,
                        osg::ref_ptr<osg::Vec4Array> colors,
                        osg::ref_ptr<osg::Vec3Array> normals,
                        osg::ref_ptr<osg::Vec2Array> texCoords);

 public:
  /**
   * @brief Get the world transform
   *
   * @return osg::ref_ptr<osg::PositionAttitudeTransform>
   */
  inline osg::ref_ptr<osg::PositionAttitudeTransform> getTransform() {
    return m_transform;
  }
  /**
   * @brief Constructor with an existing voxel array
   *
   * @param voxels
   */
  explicit VoxelMesh(VoxelArray *voxels);

  /**
   * @brief Copy constructor
   *
   * @param voxmesh
   */
  VoxelMesh(const VoxelMesh &voxmesh);

  /**
   * @brief Assignment operator
   *
   * @param voxmesh
   * @return VoxelMesh&
   */
  VoxelMesh &operator=(const VoxelMesh &voxmesh);

  /**
   * @brief Get the number of lattice sites along the X-axis
   *
   * @return int
   */
  inline int getSizeX() { return m_voxels->getSizeX(); }

  /**
   * @brief Get the number of lattice sites along the Y-axis
   *
   * @return int
   */
  inline int getSizeY() { return m_voxels->getSizeY(); }

  /**
   * @brief Get the number of lattice sites along the Z-axis
   *
   * @return int
   */
  inline int getSizeZ() { return m_voxels->getSizeZ(); }

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

  void build(VoxelMeshType::Enum type);
};
