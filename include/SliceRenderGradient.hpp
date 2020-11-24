#pragma once

#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/counting_iterator.h>

#include <iomanip>
#include <sstream>
#include <string>

#include "BillboardText.hpp"
#include "SliceRender.hpp"

#define SLICE_GRADIENT_HEIGHT 18

/**
 * @brief A 2D color gradient showing which colors represent different amounts
 * of temperature, velocity etc.
 */
class SliceRenderGradient : public SliceRender {
 private:
  osg::ref_ptr<osg::Vec3Array> m_vertices;
  real_t m_colorValues[10];
  osg::ref_ptr<BillboardText> m_labels[10];

 public:
  /**
   * @return int Number of numeric labels
   */
  inline int getNumLabels() {
    return sizeof(m_colorValues) / sizeof(m_colorValues[0]);
  }

  /**
   * @brief Get billboards for numeric labels
   *
   * @param index
   * @return osg::ref_ptr<BillboardText>
   */
  inline osg::ref_ptr<BillboardText> getLabel(int index) {
    return m_labels[index];
  }
  /**
   * @brief Set the node mask for displaying/hiding gradient and labels
   *
   * @param nm
   */
  void setNodeMask(NodeMask nm) {
    SliceRender::setNodeMask(nm);
    for (int i = 0; i < getNumLabels(); i++) getLabel(i)->setNodeMask(nm);
  }
  /**
   * @brief Resize the gradient graphic
   *
   * @param width
   * @param height
   */
  void resize(int width, int height = SLICE_GRADIENT_HEIGHT);
  /**
   * @brief Set minimum/maximum numerical values
   *
   * @param min
   * @param max
   */
  virtual void setMinMax(const real_t min, const real_t max);
  inline ~SliceRenderGradient() {}
  explicit SliceRenderGradient(int width = 1,
                               int height = SLICE_GRADIENT_HEIGHT);
};
