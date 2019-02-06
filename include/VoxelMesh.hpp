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
#include <string>
#include <utility>

#include "ColorSet.hpp"
#include "DdQq.hpp"
#include "VoxelArray.hpp"

namespace VoxelMeshType {
enum Enum { FULL, REDUCED };
}

class MeshArray {
 public:
  //! Vertices of mesh
  osg::ref_ptr<osg::Vec3Array> m_vertices;
  //! Color of each vertex
  osg::ref_ptr<osg::Vec4Array> m_colors;
  //! Plane normals
  osg::ref_ptr<osg::Vec3Array> m_normals;
  //! Texture coordinates
  osg::ref_ptr<osg::Vec2Array> m_texCoords;

  MeshArray(const MeshArray &other)
      : m_vertices(other.m_vertices),
        m_colors(other.m_colors),
        m_normals(other.m_normals),
        m_texCoords(other.m_texCoords) {
    dirty();
  }

  MeshArray()
      : m_vertices(new osg::Vec3Array()),
        m_colors(new osg::Vec4Array()),
        m_normals(new osg::Vec3Array()),
        m_texCoords(new osg::Vec2Array()) {}

  void dirty() {
    m_vertices->dirty();
    m_colors->dirty();
    m_normals->dirty();
    m_texCoords->dirty();
  }

  void insert(MeshArray *other) {
    m_vertices->insert(m_vertices->end(), other->m_vertices->begin(),
                       other->m_vertices->end());
    m_colors->insert(m_colors->end(), other->m_colors->begin(),
                     other->m_colors->end());
    m_normals->insert(m_normals->end(), other->m_normals->begin(),
                      other->m_normals->end());
    m_texCoords->insert(m_texCoords->end(), other->m_texCoords->begin(),
                        other->m_texCoords->end());
  }

  void clear() {
    m_vertices->clear();
    m_colors->clear();
    m_normals->clear();
    m_texCoords->clear();

    m_vertices->trim();
    m_colors->trim();
    m_normals->trim();
    m_texCoords->trim();
  }

  static void swap(MeshArray *f1, MeshArray *f2) {
    osg::ref_ptr<osg::Vec3Array> vertices = f1->m_vertices;
    osg::ref_ptr<osg::Vec4Array> colors = f1->m_colors;
    osg::ref_ptr<osg::Vec3Array> normals = f1->m_normals;
    osg::ref_ptr<osg::Vec2Array> texCoords = f1->m_texCoords;
    f1->m_vertices = f2->m_vertices;
    f1->m_colors = f2->m_colors;
    f1->m_normals = f2->m_normals;
    f1->m_texCoords = f2->m_texCoords;
    f2->m_vertices = vertices;
    f2->m_colors = colors;
    f2->m_normals = normals;
    f2->m_texCoords = texCoords;
  }
};

/**
 * @brief This class can build and display a mesh based on an voxel array and a
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

  MeshArray *m_arrayOrig;
  MeshArray *m_arrayTmp1;
  MeshArray *m_arrayTmp2;

  //! How to render the polygons
  osg::PolygonMode::Mode m_polyMode;

  ~VoxelMesh() { delete m_colorSet; }

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
   * @brief Construct the 3D mesh, fill the vertex, normal and color arrays.
   * Vertex reduced version from
   * https://github.com/mikolalysenko/mikolalysenko.github.com/blob/master/MinecraftMeshes2/js/greedy.js
   *
   * @param array
   */
  void buildMeshReduced(MeshArray *array);
  void buildMeshReduced(MeshArray *array, int min[3], int max[3]);

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
   * @param voxels
   */
  explicit VoxelMesh(const VoxelMesh &other);

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
