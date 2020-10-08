#pragma once

#include <osg/Geode>
#include <osg/Geometry>
#include <osg/Material>
#include <osg/PositionAttitudeTransform>
#include <osg/StateSet>
#include <osg/Vec3i>

#include <thrust/host_vector.h>

#include <iostream>

#include "SliceRenderGradient.hpp"
#include "CudaUtils.hpp"

#define HISTOGRAM_NUM_BINS 100
#define HISTOGRAM_HEIGHT 100

class HistogramMesh : public osg::Geometry {
 private:
  osg::ref_ptr<osg::Vec3Array> m_vertices;
  osg::ref_ptr<osg::PositionAttitudeTransform> m_transform;
  thrust::host_vector<real_t> m_histogram;
  float m_width, m_height;

 public:
  inline void setNodeMask(NodeMask nm) { osg::Geometry::setNodeMask(nm); }

  void update(const thrust::host_vector<real_t>& histogram);

  inline void resize(int width, int height = HISTOGRAM_HEIGHT) {
    m_width = width;
    m_height = height;
    update(m_histogram);
  }

  inline osg::ref_ptr<osg::PositionAttitudeTransform> getTransform() {
    return m_transform;
  }

  HistogramMesh();
};
