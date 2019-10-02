#pragma once

#include <osg/Geode>
#include <osg/Geometry>
#include <osg/Material>
#include <osg/PolygonMode>
#include <osg/Vec4>

#include "ColorSet.hpp"
#include "StlFile.hpp"

class StlMesh : public osg::Geode {
 protected:
  ~StlMesh() {}

 public:
  explicit StlMesh(const stl_file::StlFile& solid, const osg::Vec4 color);
};
