#pragma once

#include <osg/Array>
#include <osg/Geode>
#include <osg/Geometry>
#include <osg/Material>
#include <osg/PositionAttitudeTransform>
#include <osg/Vec3>

#include "ColorSet.hpp"
#include "Voxel.hpp"
#include "VoxelMesh.hpp"

// This class can build and display mesh based on an voxel array and color set
class VoxelContourMesh : public VoxelMesh {
 private:
  osg::ref_ptr<osg::Texture2D> m_texture;
  osg::ref_ptr<osg::Image> m_image;

 public:
  void build(VoxelMeshType::Enum type);
  explicit VoxelContourMesh(VoxelArray *voxels);
};
