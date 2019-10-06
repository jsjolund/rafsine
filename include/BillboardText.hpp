#pragma once

#include <osg/BlendFunc>
#include <osg/BoundingBox>
#include <osg/PositionAttitudeTransform>
#include <osg/Vec3d>
#include <osgText/Font>
#include <osgText/Text>

#include <string>

#include "Eigen/Geometry"

/**
 * @brief This specifies the font settings used in the OSG parts of Rafsine
 *
 */
class BillboardText : public osgText::Text {
 public:
  BillboardText();
};

osg::ref_ptr<osg::Group> createBillboardText(Eigen::Vector3i center,
                                             std::string content);
