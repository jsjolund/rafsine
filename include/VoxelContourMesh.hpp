#pragma once

#include <osg/Array>
#include <osg/Geode>
#include <osg/Geometry>
#include <osg/Material>
#include <osg/PositionAttitudeTransform>
#include <osg/Vec3>

#include "ColorSet.hpp"
#include "VoxelArray.hpp"
#include "VoxelMesh.hpp"

// This class can build and display mesh based on an voxel array and color set
class VoxelContourMesh : public VoxelMesh {
 private:
  // osg::ref_ptr<osg::Texture2D> m_texture;
  // osg::ref_ptr<osg::Image> m_image;
  void build();

 public:
  explicit VoxelContourMesh(const VoxelMesh& voxMesh);
};
