#pragma once

#include <osg/Material>
#include <osg/Node>
#include <osg/PositionAttitudeTransform>
#include <osg/StateSet>
#include <osgDB/ReadFile>

class AxesMesh : public osg::PositionAttitudeTransform {
 public:
  AxesMesh();
  void resize(int width, int height);
};
