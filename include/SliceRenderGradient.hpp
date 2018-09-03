#pragma once

#include <osg/BoundingBox>
#include <osgText/Font>
#include <osgText/Text>

#include <sstream>
#include <string>
#include <iomanip>

#include "SliceRender.hpp"

class SliceRenderGradient : public SliceRender
{
private:
  thrust::device_vector<real> m_gradient;
  real m_colorValues[10];
  osg::ref_ptr<osgText::Text> m_labels[10];

public:
  inline int getNumLabels() { return sizeof(m_colorValues) / sizeof(m_colorValues[0]); }
  inline osg::ref_ptr<osgText::Text> *getLabels() { return m_labels; }
  void resize(unsigned int width, unsigned int height);
  virtual void setMinMax(real min, real max);
  SliceRenderGradient(unsigned int width, unsigned int height);
};