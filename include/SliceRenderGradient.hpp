#pragma once

#include <iomanip>
#include <sstream>
#include <string>

#include "CFDSceneText.hpp"
#include "SliceRender.hpp"

#define SLICE_GRADIENT_HEIGHT 18

/**
 * @brief A 2D color gradient showing which colors represent different amounts
 * of temperature, velocity etc.
 *
 */
class SliceRenderGradient : public SliceRender {
 private:
  osg::ref_ptr<osg::Vec3Array> m_vertices;
  thrust::device_vector<real> m_gradient;
  real m_colorValues[10];
  osg::ref_ptr<osgText::Text> m_labels[10];

 public:
  inline int getNumLabels() {
    return sizeof(m_colorValues) / sizeof(m_colorValues[0]);
  }
  inline osg::ref_ptr<osgText::Text> getLabel(int index) {
    return m_labels[index];
  }
  void resize(unsigned int width, unsigned int height = SLICE_GRADIENT_HEIGHT);
  virtual void setMinMax(real min, real max);
  inline ~SliceRenderGradient(){};
  SliceRenderGradient(unsigned int width = 1,
                      unsigned int height = SLICE_GRADIENT_HEIGHT);
};
