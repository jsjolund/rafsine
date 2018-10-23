#pragma once

#include <osg/Material>
#include <osg/Node>
#include <osg/PositionAttitudeTransform>
#include <osg/StateSet>
#include <osgDB/ReadFile>

/**
 * @brief A 3d model showing the x,y,z-axis on the HUD
 *
 */
class AxesMesh : public osg::PositionAttitudeTransform {
 public:
  AxesMesh();
  /**
   * @brief Sets the size of the axis in the HUD
   *
   * @param width
   * @param height
   */
  void resize(int width, int height);
};
