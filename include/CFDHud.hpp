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
 private:
  // The HUD projection matrix
  osg::ref_ptr<osg::Projection> m_projectionMatrix = new osg::Projection;

 public:
  /**
   * @brief Get the HUD projection matrix
   * @return osg::ref_ptr<osg::Projection>
   */
  inline osg::ref_ptr<osg::Projection> getProjectionMatrix() {
    return m_projectionMatrix;
  }
  /**
   * @brief Construct a new HUD
   *
   * @param width
   * @param height
   */
  CFDHud(int width, int height);
  /**
   * @brief Scale the HUD projection matrix to fit the screen
   *
   * @param width
   * @param height
   */
  void resize(int width, int height);
};
