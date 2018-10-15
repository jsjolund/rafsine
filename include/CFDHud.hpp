#pragma once

#include <osg/Geode>
#include <osg/Matrix>
#include <osg/MatrixTransform>
#include <osg/Projection>
#include <osg/Vec3d>
#include <osgDB/ReadFile>
#include <osgText/Font>
#include <osgText/Text>

#include <limits.h>

/**
 * @brief A helper class for drawing HUD objects on the screen
 * 
 */
class CFDHud : public osg::Geode {
 public:
  osg::ref_ptr<osg::Projection> m_projectionMatrix = new osg::Projection;

  CFDHud(int width, int height);
  void resize(int width, int height);
};
