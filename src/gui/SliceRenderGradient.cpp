#include "SliceRenderGradient.hpp"

SliceRenderGradient::SliceRenderGradient(unsigned int width,
                                         unsigned int height)
    : SliceRender(SliceRenderAxis::GRADIENT,
                  width,
                  height,
                  NULL,
                  osg::Vec3i(width, height, 0)),
      m_gradient(width * height)
{
  m_plot3d = (real *)thrust::raw_pointer_cast(&(m_gradient)[0]);

  osg::ref_ptr<osgText::Font> font = osgText::readFontFile("fonts/arial.ttf");
  font->setMinFilterHint(osg::Texture::LINEAR_MIPMAP_LINEAR);
  font->setMagFilterHint(osg::Texture::LINEAR);
  font->setMaxAnisotropy(16.0f);

  // Create text labels for gradient ticks
  for (int i = 0; i < getNumLabels(); i++)
  {
    m_colorValues[i] = 0.0f;

    osg::ref_ptr<osgText::Text> label = new osgText::Text;
    m_labels[i] = label;

    label->setFont(font);
    label->setFontResolution(80, 80);
    label->setColor(osg::Vec4(1.0f, 1.0f, 1.0f, 1.0f));
    label->setBackdropType(osgText::Text::OUTLINE);
    label->setBackdropOffset(0.1f);
    label->setBackdropColor(osg::Vec4(0.0f, 0.0f, 0.0f, 0.3f));
    label->setBackdropImplementation(osgText::Text::DEPTH_RANGE);
    label->setShaderTechnique(osgText::ALL_FEATURES);
    label->setCharacterSize(14);
    label->setAxisAlignment(osgText::Text::SCREEN);
    label->setDataVariance(osg::Object::DYNAMIC);
    osg::StateSet *stateSet = label->getOrCreateStateSet();
    stateSet->setMode(GL_BLEND, osg::StateAttribute::ON);
  }
  resize(width, height);
}

void SliceRenderGradient::resize(unsigned int width, unsigned int height)
{
  // Resize the gradient quad
  osg::ref_ptr<osg::Vec3Array> vertices = new osg::Vec3Array;
  vertices->push_back(osg::Vec3(0, 0, -1));
  vertices->push_back(osg::Vec3(width, 0, -1));
  vertices->push_back(osg::Vec3(width, height, -1));
  vertices->push_back(osg::Vec3(0, height, -1));
  setVertexArray(vertices);

  // Calculate label positions
  float labelWidth = m_labels[0]->computeBound().radius() * 2;
  float labelSpacing = ((float)width) / (getNumLabels() - 1);
  float vOffset = 4;
  m_labels[0]->setPosition(osg::Vec3(0, vOffset, -1));
  m_labels[getNumLabels() - 1]->setPosition(
      osg::Vec3(width - labelWidth, vOffset, -1));
  for (int i = 1; i < getNumLabels() - 1; i++)
    m_labels[i]->setPosition(
        osg::Vec3(i * labelSpacing - labelWidth / 2, vOffset, -1));
}

void SliceRenderGradient::setMinMax(real min, real max)
{
  m_min = min;
  m_max = max;
  // Calculate ticks between min and max value
  real Dx = (m_max - m_min) / (real)(m_voxSize.x() * m_voxSize.y() - 1);
  thrust::transform(thrust::make_counting_iterator(m_min / Dx),
                    thrust::make_counting_iterator((m_max + 1.f) / Dx),
                    thrust::make_constant_iterator(Dx),
                    m_gradient.begin(),
                    thrust::multiplies<real>());
  int numTicks = sizeof(m_colorValues) / sizeof(m_colorValues[0]);
  Dx = (m_max - m_min) / (real)(numTicks - 1);
  for (int i = 0; i < numTicks - 1; i++)
    m_colorValues[i] = m_min + Dx * i;
  m_colorValues[numTicks - 1] = m_max;

  // Update the text labels
  for (int i = 0; i < getNumLabels(); i++)
  {
    osg::ref_ptr<osgText::Text> label = m_labels[i];
    std::stringstream ss;
    ss << std::fixed << std::setprecision(2) << m_colorValues[i];
    label->setText(ss.str());
  }
}