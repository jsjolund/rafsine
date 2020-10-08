#include "SliceRenderGradient.hpp"

SliceRenderGradient::SliceRenderGradient(int width, int height)
    : SliceRender(D3Q4::ORIGIN, width, height, osg::Vec3i(width, height, 0)),
      m_vertices(new osg::Vec3Array()) {
  // Create text labels for gradient ticks
  for (int i = 0; i < getNumLabels(); i++) {
    m_colorValues[i] = 0.0f;
    m_labels[i] = new BillboardText();
  }
  resize(width, height);
}

void SliceRenderGradient::resize(int width, int height) {
  int depth = 0;
  // Resize the gradient quad
  m_vertices->clear();
  m_vertices->resize(0);
  m_vertices->push_back(osg::Vec3(0, 0, depth));
  m_vertices->push_back(osg::Vec3(width, 0, depth));
  m_vertices->push_back(osg::Vec3(width, height, depth));
  m_vertices->push_back(osg::Vec3(0, height, depth));
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

  m_transform->setPivotPoint(osg::Vec3d(width / 2, height, 0));
  m_transform->setAttitude(osg::Quat(osg::PI, osg::Vec3d(0, 0, 1)));
  m_transform->setPosition(osg::Vec3d(width / 2, 0, 0));
}

void SliceRenderGradient::setMinMax(const real_t min, const real_t max) {
  m_min = min;
  m_max = max;
  //
  real_t Dx = (m_max - m_min) / (m_plot3dSize.x() * m_plot3dSize.y() - 1);
  if (m_min != m_max) {
    // Draw the gradient plot. Transform will generate a vector starting with
    // min/dx*dx, adding dx to each element and ending with (max+dx)/dx*dx-dx.
    // TODO(sometimes the last element is not set correctly, filling fixes it)
    thrust::fill(m_plot2d.begin(), m_plot2d.end(), m_max);
    thrust::transform(thrust::make_counting_iterator(m_min / Dx),
                      thrust::make_counting_iterator((m_max + Dx) / Dx),
                      thrust::make_constant_iterator(Dx), m_plot2d.begin(),
                      thrust::multiplies<real_t>());
  } else {
    thrust::fill(m_plot2d.begin(), m_plot2d.end(), m_min);
  }

  // Calculate the values of labels
  Dx = (m_max - m_min) / (getNumLabels() - 1);
  for (int i = 0; i < getNumLabels() - 1; i++)
    m_colorValues[i] = m_min + Dx * i;
  m_colorValues[getNumLabels() - 1] = m_max;

  // Update the text labels
  for (int i = 0; i < getNumLabels(); i++) {
    osg::ref_ptr<BillboardText> label = m_labels[i];
    std::stringstream ss;
    ss << std::fixed << std::setprecision(2) << m_colorValues[i];
    label->setText(ss.str());
  }
}
