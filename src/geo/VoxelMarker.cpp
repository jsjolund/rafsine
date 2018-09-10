#include "VoxelMarker.hpp"

VoxelMarker::VoxelMarker()
{
  osg::Box *box = new osg::Box(osg::Vec3(0, 0, 0), 1.1f);
  addDrawable(new osg::ShapeDrawable(box));

  osg::StateSet *stateSet = getOrCreateStateSet();
  stateSet->setMode(GL_DEPTH_TEST, osg::StateAttribute::ON);

  osg::ref_ptr<osg::Material> mat = new osg::Material();
  mat->setEmission(osg::Material::Face::FRONT_AND_BACK,
                   osg::Vec4f(1.0f, 1.0f, 1.0f, 1.0f));
  mat->setColorMode(osg::Material::ColorMode::EMISSION);

  stateSet->setAttribute(mat.get(), osg::StateAttribute::Values::ON);

  m_transform = new osg::PositionAttitudeTransform();
  m_transform->addChild(this);

  // Text should be in a separate geode
  m_text = new osgText::Text;
  osg::ref_ptr<osgText::Font> font = osgText::readFontFile("fonts/arial.ttf");
  font->setMinFilterHint(osg::Texture::LINEAR_MIPMAP_LINEAR);
  font->setMagFilterHint(osg::Texture::LINEAR);
  font->setMaxAnisotropy(16.0f);
  m_text->setFont(font);
  m_text->setCharacterSize(14);
  m_text->setFontResolution(80, 80);
  m_text->setColor(osg::Vec4(1.0f, 1.0f, 1.0f, 1.0f));

  m_text->setBackdropType(osgText::Text::OUTLINE);
  m_text->setBackdropOffset(0.15f);
  m_text->setBackdropColor(osg::Vec4(0.0f, 0.0f, 0.0f, 0.7f));
  m_text->setBackdropImplementation(osgText::Text::DEPTH_RANGE);

  m_text->setBoundingBoxColor(osg::Vec4(0.0f, 0.0f, 0.0f, 0.5f));

  m_text->setShaderTechnique(osgText::ALL_FEATURES);
  m_text->setAxisAlignment(osgText::Text::SCREEN);
  m_text->setAlignment(osgText::Text::LEFT_TOP);
  m_text->setCharacterSizeMode(osgText::Text::SCREEN_COORDS);
  m_text->setDrawMode(osgText::Text::TEXT |
                      osgText::Text::ALIGNMENT |
                      osgText::Text::FILLEDBOUNDINGBOX);
  m_text->setDataVariance(osg::Object::DYNAMIC);
  osg::StateSet *textStateSet = m_text->getOrCreateStateSet();
  textStateSet->setAttributeAndModes(new osg::BlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA));
  textStateSet->setMode(GL_BLEND, osg::StateAttribute::ON);

  m_transform->addChild(m_text);
}