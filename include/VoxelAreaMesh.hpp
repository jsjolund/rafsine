#pragma once

#include <osg/Array>
#include <osg/Geode>
#include <osg/Geometry>
#include <osg/Material>
#include <osg/PositionAttitudeTransform>
#include <osg/ShapeDrawable>
#include <osg/TexMat>
#include <osg/Texture2D>
#include <osg/Vec3>
#include <osg/Vec4>
#include <osgDB/ReadFile>

#include <glm/vec3.hpp>

#include "ColorSet.hpp"
#include "MeshArray.hpp"

class VoxelAreaMesh : public osg::Geometry {
 private:
  osg::ref_ptr<osg::Texture2D> m_texture;
  osg::ref_ptr<osg::Image> m_image;
  ColorSet m_colorSet;
  MeshArray m_array;

 public:
  MeshArray createBox(glm::ivec3 min, glm::ivec3 max, osg::Vec4 color);
  VoxelAreaMesh(glm::ivec3 min, glm::ivec3 max);
};
