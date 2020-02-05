#include "VoxelMarker.hpp"

VoxelMarker::VoxelMarker() {
  addDrawable(new osg::ShapeDrawable(new osg::Box(osg::Vec3(0, 0, 0), 1.1f)));

  osg::StateSet* stateset = getOrCreateStateSet();
  stateset->setMode(GL_DEPTH_TEST, osg::StateAttribute::ON);

  osg::ref_ptr<osg::Material> mat = new osg::Material();
  mat->setEmission(osg::Material::Face::FRONT_AND_BACK,
                   osg::Vec4f(1.0f, 1.0f, 1.0f, 1.0f));
  mat->setColorMode(osg::Material::ColorMode::EMISSION);

  stateset->setAttribute(mat.get(), osg::StateAttribute::Values::ON);

  m_transform = new osg::PositionAttitudeTransform();
  m_transform->addChild(this);

  // Text should be in a separate geode
  m_text = new BillboardText();
  m_text->setBoundingBoxColor(osg::Vec4(0.0f, 0.0f, 0.0f, 0.5f));
  m_text->setCharacterSizeMode(osgText::Text::SCREEN_COORDS);
  m_text->setDrawMode(osgText::Text::TEXT | osgText::Text::ALIGNMENT |
                      osgText::Text::FILLEDBOUNDINGBOX);
  m_text->setAlignment(osgText::Text::LEFT_TOP);
  m_transform->addChild(m_text);
}
