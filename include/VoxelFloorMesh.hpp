#pragma once

#include <osg/Geode>
#include <osg/Geometry>
#include <osg/Image>
#include <osg/PositionAttitudeTransform>
#include <osg/StateSet>
#include <osg/Texture2D>

#include <cuda.h>
#include <cuda_gl_interop.h>

#include <omp.h>

#include <osg/Vec3i>

#include "VoxelArray.hpp"

/**
 * @brief Displays the texture of the floor
 *
 */
class VoxelFloorMesh : public osg::Geometry {
 private:
  int m_width, m_height;
  // Image to draw on texture
  osg::ref_ptr<osg::Image> m_image;
  // World transform
  osg::ref_ptr<osg::PositionAttitudeTransform> m_transform;
  osg::ref_ptr<osg::Texture2D> m_texture;

  void set(int x, int y, osg::Vec3i color);

 public:
  explicit VoxelFloorMesh(std::shared_ptr<VoxelArray> voxels);
  inline osg::ref_ptr<osg::PositionAttitudeTransform> getTransform() {
    return m_transform;
  }
};
