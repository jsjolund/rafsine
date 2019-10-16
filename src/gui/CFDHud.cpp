#include "CFDHud.hpp"

CFDHud::CFDHud(int width, int height)
    : osg::Geode(), m_projectionMatrix(new osg::Projection) {
  // Set up the matrices for HUD projection
  m_projectionMatrix->setMatrix(osg::Matrix::ortho2D(0, width, 0, height));
  osg::ref_ptr<osg::MatrixTransform> modelViewMatrix = new osg::MatrixTransform;
  modelViewMatrix->setMatrix(osg::Matrix::identity());
  modelViewMatrix->setReferenceFrame(osg::Transform::ABSOLUTE_RF);
  m_projectionMatrix->addChild(modelViewMatrix);
  modelViewMatrix->addChild(this);

  // Create and set up a state set using the texture from above:
  osg::StateSet* stateset = getOrCreateStateSet();
  // stateset->setTextureAttributeAndModes(0, HUDTexture,
  // osg::StateAttribute::ON);

  // For this state set, turn blending on (so alpha texture looks right)
  stateset->setMode(GL_BLEND, osg::StateAttribute::ON);

  stateset->setRenderingHint(osg::StateSet::TRANSPARENT_BIN);

  // Disable depth testing so geometry is draw regardless of depth values
  // of geometry already draw.

  stateset->setMode(GL_DEPTH_TEST, osg::StateAttribute::OFF);
  stateset->setRenderBinDetails(INT_MAX, "RenderBin");
}

void CFDHud::resize(int width, int height) {
  m_projectionMatrix->setMatrix(osg::Matrix::ortho2D(0, width, 0, height));
}
