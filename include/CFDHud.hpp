#pragma once

#include <osg/Projection>
#include <osg/Geode>
#include <osg/Vec3d>
#include <osg/Matrix>
#include <osg/MatrixTransform>
#include <osgText/Font>
#include <osgText/Text>
#include <osgDB/ReadFile>

class CFDHud : public osg::Geode
{
public:
  osg::Projection *m_projectionMatrix = new osg::Projection;

  CFDHud(int width, int height);
  void resize(int width, int height);
};