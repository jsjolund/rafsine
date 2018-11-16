#include "SliceRenderGradient.hpp"

SliceRenderGradient::SliceRenderGradient(unsigned int width,
                                         unsigned int height)
    : SliceRender(SliceRenderAxis::GRADIENT, width, height, NULL,
                  osg::Vec3i(width, height, 0)),
      m_gradient(width * height),
      m_vertices(new osg::Vec3Array()) {
  m_plot3d =
      reinterpret_cast<real *>(thrust::raw_pointer_cast(&(m_gradient)[0]));
  // Create text labels for gradient ticks
  for (int i = 0; i < getNumLabels(); i++) {
    m_colorValues[i] = 0.0f;
    m_labels[i] = new CFDSceneText();
  }
  resize(width, height);
  // m_transform->setPivotPoint(osg::Vec3d(width/2,height/2,0));
  // m_transform->setAttitude(osg::Quat(osg::PI, osg::Vec3d(0, 0, 1)));
}

void SliceRenderGradient::resize(unsigned int width, unsigned int height) {
  // Resize the gradient quad
  m_vertices->clear();
  m_vertices->resize(0);
  m_vertices->push_back(osg::Vec3(0, 0, -1));
  m_vertices->push_back(osg::Vec3(width, 0, -1));
  m_vertices->push_back(osg::Vec3(width, height, -1));
  m_vertices->push_back(osg::Vec3(0, height, -1));
  setVertexArray(m_vertices);

  // Calculate label positions
  float labelWidth = m_labels[0]->computeBound().radius() * 2;
  float labelSpacing = static_cast<float>(width) / (getNumLabels() - 1);
  float vOffset = 4;
  m_labels[0]->setPosition(osg::Vec3(0, vOffset, -1));
  m_labels[getNumLabels() - 1]->setPosition(
      osg::Vec3(width - labelWidth, vOffset, -1));
  for (int i = 1; i < getNumLabels() - 1; i++)
    m_labels[i]->setPosition(
        osg::Vec3(i * labelSpacing - labelWidth / 2, vOffset, -1));
}

void SliceRenderGradient::setMinMax(real min, real max) {
  m_min = min;
  m_max = max;
  // Calculate ticks between min and max value
  real Dx = (m_max - m_min) / (real)(m_voxSize.x() * m_voxSize.y() - 1);
  if (m_min != m_max) {
    // Draw the gradient plot
    thrust::transform(thrust::make_counting_iterator(m_min / Dx),
                      thrust::make_counting_iterator((m_max + 1.f) / Dx),
                      thrust::make_constant_iterator(Dx), m_gradient.begin(),
                      thrust::multiplies<real>());
  } else {
    thrust::fill(m_gradient.begin(), m_gradient.end(), m_min);
  }
  // Calculate the values of labels
  Dx = (m_max - m_min) / (real)(getNumLabels() - 1);
  for (int i = 0; i < getNumLabels() - 1; i++)
    m_colorValues[i] = m_min + Dx * i;
  m_colorValues[getNumLabels() - 1] = m_max;

  // Update the text labels
  for (int i = 0; i < getNumLabels(); i++) {
    osg::ref_ptr<osgText::Text> label = m_labels[i];
    std::stringstream ss;
    ss << std::fixed << std::setprecision(2) << m_colorValues[i];
    label->setText(ss.str());
  }
}
