#pragma once

#include <osg/Geode>
#include <osg/Geometry>
#include <osg/Material>
#include <osg/PolygonMode>
#include <osg/Vec4>

#include "ColorSet.hpp"
#include "StlMesh.hpp"

class StlModel : public osg::Geode {
 protected:
  ~StlModel() {}

 public:
  explicit StlModel(const stl_mesh::StlMesh& solid, const osg::Vec4 color);
};
