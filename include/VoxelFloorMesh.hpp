#pragma once

#include <osg/Image>
#include <osg/Geometry>
#include <osg/StateSet>
#include <osg/Texture2D>
#include <osg/PositionAttitudeTransform>
#include <osg/Geode>

#include <cuda.h>
#include <cuda_gl_interop.h>

#include "Voxel.hpp"

class VoxelFloorMesh : public osg::Geometry
{
private:
  int m_width, m_height;
  // Image to draw on texture
  osg::ref_ptr<osg::Image> m_image;
  // Voxels
  VoxelArray *m_voxels;
  // World transform
  osg::ref_ptr<osg::PositionAttitudeTransform> m_transform;
  osg::ref_ptr<osg::Texture2D> m_texture;

public:
  VoxelFloorMesh(VoxelArray *voxels);
  virtual void drawImplementation(osg::RenderInfo &renderInfo) const;
  inline osg::ref_ptr<osg::PositionAttitudeTransform> getTransform() { return m_transform; }
};