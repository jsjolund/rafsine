#pragma once

#include <osg/Material>
#include <osg/Shape>
#include <osgText/Text>
#include <osgText/Font>
#include <osg/Geode>
#include <osg/ShapeDrawable>
#include <osg/BlendFunc>
#include <osg/PositionAttitudeTransform>

class VoxelMarker : public osg::Geode
{
private:
  osg::ref_ptr<osg::PositionAttitudeTransform> m_transform;
  osg::ref_ptr<osgText::Text> m_text;

public:
  VoxelMarker();

  inline osg::ref_ptr<osgText::Text> getLabel() { return m_text; }
  inline osg::ref_ptr<osg::PositionAttitudeTransform> getTransform() { return m_transform; }
};