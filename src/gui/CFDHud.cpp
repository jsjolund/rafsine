#include "CFDHud.hpp"

CFDHud::CFDHud(int width, int height)
    : osg::Geode(),
      m_projectionMatrix(new osg::Projection)
{
  m_projectionMatrix->setMatrix(osg::Matrix::ortho2D(0, width, 0, height));
  osg::MatrixTransform *modelViewMatrix = new osg::MatrixTransform;
  modelViewMatrix->setMatrix(osg::Matrix::identity());
  modelViewMatrix->setReferenceFrame(osg::Transform::ABSOLUTE_RF);
  m_projectionMatrix->addChild(modelViewMatrix);
  modelViewMatrix->addChild(this);

  // Create and set up a state set using the texture from above:
  osg::StateSet *stateSet = getOrCreateStateSet();
  // stateSet->setTextureAttributeAndModes(0, HUDTexture, osg::StateAttribute::ON);

  // For this state set, turn blending on (so alpha texture looks right)
  stateSet->setMode(GL_BLEND, osg::StateAttribute::ON);

  // Disable depth testing so geometry is draw regardless of depth values
  // of geometry already draw.
  stateSet->setMode(GL_DEPTH_TEST, osg::StateAttribute::OFF);
  stateSet->setRenderingHint(osg::StateSet::TRANSPARENT_BIN);

  // Need to make sure this geometry is draw last. RenderBins are handled
  // in numerical order so set bin number to 11
  stateSet->setRenderBinDetails(11, "RenderBin");
}

void CFDHud::resize(int width, int height)
{
  m_projectionMatrix->setMatrix(osg::Matrix::ortho2D(0, width, 0, height));
}