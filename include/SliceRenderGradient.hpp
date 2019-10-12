#pragma once

#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/constant_iterator.h>

#include <iomanip>
#include <sstream>
#include <string>

#include "BillboardText.hpp"
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
  real m_colorValues[10];
  osg::ref_ptr<BillboardText> m_labels[10];

 public:
  inline int getNumLabels() {
    return sizeof(m_colorValues) / sizeof(m_colorValues[0]);
  }
  inline osg::ref_ptr<BillboardText> getLabel(int index) {
    return m_labels[index];
  }
  void setNodeMask(NodeMask nm) {
    SliceRender::setNodeMask(nm);
    for (int i = 0; i < getNumLabels(); i++) getLabel(i)->setNodeMask(nm);
  }
  void resize(int width, int height = SLICE_GRADIENT_HEIGHT);
  virtual void setMinMax(const real min, const real max);
  inline ~SliceRenderGradient() {}
  explicit SliceRenderGradient(int width = 1,
                               int height = SLICE_GRADIENT_HEIGHT);
};
