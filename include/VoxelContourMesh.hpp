#pragma once

#include <osg/Array>
#include <osg/BlendFunc>
#include <osg/Geode>
#include <osg/Geometry>
#include <osg/Material>
#include <osg/PolygonOffset>
#include <osg/PositionAttitudeTransform>
#include <osg/Vec3>

#include "ColorSet.hpp"
#include "VoxelArray.hpp"
#include "VoxelMesh.hpp"

/**
 * @brief Creates a mesh displaying outline/contour lines of a voxel mesh
 */
class VoxelContourMesh : public VoxelMesh {
 private:
  void build();

 public:
  /**
   * @brief Construct contour mesh of voxel mesh
   *
   * @param voxMesh
   */
  explicit VoxelContourMesh(const VoxelMesh& voxMesh);
};
