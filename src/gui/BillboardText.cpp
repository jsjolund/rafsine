#include "BillboardText.hpp"

BillboardText::BillboardText() : osgText::Text() {
  osg::ref_ptr<osgText::Font> font = osgText::readFontFile("fonts/arial.ttf");
  font->setMinFilterHint(osg::Texture::LINEAR_MIPMAP_LINEAR);
  font->setMagFilterHint(osg::Texture::LINEAR);
  font->setMaxAnisotropy(16.0f);
  setFont(font);

  setCharacterSize(14);
  setFontResolution(80, 80);
  setColor(osg::Vec4(1.0f, 1.0f, 1.0f, 1.0f));
  setBackdropType(osgText::Text::OUTLINE);
  setBackdropOffset(0.15f);
  setBackdropColor(osg::Vec4(0.0f, 0.0f, 0.0f, 0.7f));
  setBackdropImplementation(osgText::Text::DEPTH_RANGE);
  setShaderTechnique(osgText::ALL_FEATURES);
  setAxisAlignment(osgText::Text::SCREEN);
  setDataVariance(osg::Object::DYNAMIC);

  osg::StateSet* stateset = getOrCreateStateSet();
  stateset->setAttributeAndModes(
      new osg::BlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA));
  stateset->setMode(GL_BLEND, osg::StateAttribute::ON);
  stateset->setMode(GL_DEPTH_TEST, osg::StateAttribute::OFF);
  stateset->setRenderBinDetails(INT_MAX - 1, "RenderBin");
}

osg::ref_ptr<osg::PositionAttitudeTransform> createBillboardText(
    Eigen::Vector3i center,
    std::string content) {
  osg::ref_ptr<osg::PositionAttitudeTransform> transform =
      new osg::PositionAttitudeTransform();
  osg::ref_ptr<osgText::Text> text = new BillboardText();
  text->setBoundingBoxColor(osg::Vec4(0.0f, 0.0f, 0.0f, 0.5f));
  text->setCharacterSizeMode(osgText::Text::SCREEN_COORDS);
  text->setDrawMode(osgText::Text::TEXT |
                    // osgText::Text::ALIGNMENT |
                    osgText::Text::FILLEDBOUNDINGBOX);
  text->setAlignment(osgText::Text::LEFT_TOP);
  transform->addChild(text);
  transform->setPosition(osg::Vec3d(center.x(), center.y(), center.z()));
  // addChild(transform);
  text->setText(content);
  return transform;
}
