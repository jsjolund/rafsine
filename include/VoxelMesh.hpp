#pragma once

#include <osg/Array>
#include <osg/Geode>
#include <osg/Geometry>
#include <osg/Material>
#include <osg/PositionAttitudeTransform>
#include <osg/Vec3>

#include <string>

#include "ColorSet.hpp"
#include "Voxel.hpp"

/**
 * @brief  This class can build and display a mesh based on an voxel array and a
 * color set
 *
 */
class VoxelMesh : public osg::Geometry {
 protected:
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

  ~VoxelMesh() { delete m_colorSet; }

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
   * @brief Construct the 3D mesh, fill the vertex, normals and color arrays
   *
   * @param voxMin
   * @param voxMax
   */
  void buildMesh(osg::Vec3i voxMin, osg::Vec3i voxMax);
};
