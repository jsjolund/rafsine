#pragma once

#include <osg/BlendFunc>
#include <osg/BoundingBox>
#include <osg/PositionAttitudeTransform>
#include <osg/Vec3d>
#include <osgText/Font>
#include <osgText/Text>

#include <glm/vec3.hpp>

#include <string>

/**
 * @brief This specifies the font settings used in the OSG parts of Rafsine
 *
 */
class BillboardText : public osgText::Text {
 public:
  BillboardText();
};

osg::ref_ptr<osg::Group> createBillboardText(glm::ivec3 center,
                                             std::string content);
