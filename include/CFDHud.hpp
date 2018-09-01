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
  // Text instance that wil show up in the HUD:
  osgText::Text *textOne = new osgText::Text();
  // Text instance for a label that will follow the tank:
  osgText::Text *tankLabel = new osgText::Text();
  // Projection node for defining view frustrum for HUD:
  osg::Projection *HUDProjectionMatrix = new osg::Projection;

  CFDHud();
  void resize(int width, int height);
};