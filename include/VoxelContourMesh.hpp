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
 public:
  void buildMesh();
  explicit VoxelContourMesh(VoxelArray *voxels);
};
