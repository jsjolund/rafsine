#pragma once

#include <osg/Vec3>
#include <osg/Array>
#include <osg/Geometry>
#include <osg/Material>
#include <osg/PositionAttitudeTransform>
#include <osg/Geode>

#include "VoxelMesh.hpp"
#include "Voxel.hpp"
#include "ColorSet.hpp"

//This class can build and display a mesh based on an voxel array and a color set
class VoxelDotMesh : public VoxelMesh
{
public:
  void buildMesh();
  VoxelDotMesh(const VoxelArray &voxels);
};