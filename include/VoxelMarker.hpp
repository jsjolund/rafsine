#pragma once

#include <osg/Geode>
#include <osg/Material>
#include <osg/PositionAttitudeTransform>
#include <osg/Shape>
#include <osg/ShapeDrawable>

#include "CFDSceneText.hpp"

/**
 * @brief Highlights a voxel after a mouse click
 * 
 */
class VoxelMarker : public osg::Geode {
 private:
  osg::ref_ptr<osg::PositionAttitudeTransform> m_transform;
  osg::ref_ptr<osgText::Text> m_text;

 public:
  VoxelMarker();

  inline osg::ref_ptr<osgText::Text> getLabel() { return m_text; }
  inline osg::ref_ptr<osg::PositionAttitudeTransform> getTransform() {
    return m_transform;
  }
};
