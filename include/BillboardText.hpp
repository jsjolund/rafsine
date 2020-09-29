#pragma once

#include <osg/BlendFunc>
#include <osg/BoundingBox>
#include <osg/PositionAttitudeTransform>
#include <osg/Vec3d>
#include <osgText/Font>
#include <osgText/Text>

#include <string>

#include "Vector3.hpp"

/**
 * @brief This specifies the font settings used in the OSG parts of Rafsine
 *
 */
class BillboardText : public osgText::Text {
 public:
  BillboardText();
};

osg::ref_ptr<osg::PositionAttitudeTransform> createBillboardText(
    vector3<int> center,
    std::string content);
