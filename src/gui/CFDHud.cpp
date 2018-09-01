#include "CFDHud.hpp"

CFDHud::CFDHud()
    : osg::Geode()
{
  HUDProjectionMatrix->setMatrix(osg::Matrix::ortho2D(0, 1024, 0, 768));
  osg::MatrixTransform *HUDModelViewMatrix = new osg::MatrixTransform;
  HUDModelViewMatrix->setMatrix(osg::Matrix::identity());
  HUDModelViewMatrix->setReferenceFrame(osg::Transform::ABSOLUTE_RF);
  HUDProjectionMatrix->addChild(HUDModelViewMatrix);

  // Add the Geometry node to contain HUD geometry as a child of the
  // HUD model view matrix.
  HUDModelViewMatrix->addChild(this);

  // Set up geometry for the HUD and add it to the HUD
  osg::Geometry *HUDBackgroundGeometry = new osg::Geometry();

  osg::Vec3Array *HUDBackgroundVertices = new osg::Vec3Array;
  HUDBackgroundVertices->push_back(osg::Vec3(0, 0, -1));
  HUDBackgroundVertices->push_back(osg::Vec3(1024, 0, -1));
  HUDBackgroundVertices->push_back(osg::Vec3(1024, 200, -1));
  HUDBackgroundVertices->push_back(osg::Vec3(0, 200, -1));

  osg::DrawElementsUInt *HUDBackgroundIndices =
      new osg::DrawElementsUInt(osg::PrimitiveSet::POLYGON, 0);
  HUDBackgroundIndices->push_back(0);
  HUDBackgroundIndices->push_back(1);
  HUDBackgroundIndices->push_back(2);
  HUDBackgroundIndices->push_back(3);

  osg::Vec4Array *HUDcolors = new osg::Vec4Array;
  HUDcolors->push_back(osg::Vec4(0.8f, 0.8f, 0.8f, 0.8f));

  osg::Vec2Array *texcoords = new osg::Vec2Array(4);
  (*texcoords)[0].set(0.0f, 0.0f);
  (*texcoords)[1].set(1.0f, 0.0f);
  (*texcoords)[2].set(1.0f, 1.0f);
  (*texcoords)[3].set(0.0f, 1.0f);

  HUDBackgroundGeometry->setTexCoordArray(0, texcoords);
  osg::Texture2D *HUDTexture = new osg::Texture2D;
  HUDTexture->setDataVariance(osg::Object::DYNAMIC);
  osg::Image *hudImage;
  hudImage = osgDB::readImageFile("assets/logo.jpg");
  HUDTexture->setImage(hudImage);
  osg::Vec3Array *HUDnormals = new osg::Vec3Array;
  HUDnormals->push_back(osg::Vec3(0.0f, 0.0f, 1.0f));
  HUDBackgroundGeometry->setNormalArray(HUDnormals);
  HUDBackgroundGeometry->setNormalBinding(osg::Geometry::BIND_OVERALL);
  HUDBackgroundGeometry->addPrimitiveSet(HUDBackgroundIndices);
  HUDBackgroundGeometry->setVertexArray(HUDBackgroundVertices);
  HUDBackgroundGeometry->setColorArray(HUDcolors);
  HUDBackgroundGeometry->setColorBinding(osg::Geometry::BIND_OVERALL);

  this->addDrawable(HUDBackgroundGeometry);

  // Create and set up a state set using the texture from above:
  osg::StateSet *HUDStateSet = new osg::StateSet();
  this->setStateSet(HUDStateSet);
  HUDStateSet->setTextureAttributeAndModes(0, HUDTexture, osg::StateAttribute::ON);

  // For this state set, turn blending on (so alpha texture looks right)
  HUDStateSet->setMode(GL_BLEND, osg::StateAttribute::ON);

  // Disable depth testing so geometry is draw regardless of depth values
  // of geometry already draw.
  HUDStateSet->setMode(GL_DEPTH_TEST, osg::StateAttribute::OFF);
  HUDStateSet->setRenderingHint(osg::StateSet::TRANSPARENT_BIN);

  // Need to make sure this geometry is draw last. RenderBins are handled
  // in numerical order so set bin number to 11
  HUDStateSet->setRenderBinDetails(11, "RenderBin");

  // Add the text (Text class is derived from drawable) to the geode:
  this->addDrawable(textOne);

  // Set up the parameters for the text we'll add to the HUD:
  textOne->setCharacterSize(25);
  textOne->setFont("fonts/arial.ttf");
  textOne->setText("Not so good");
  textOne->setAxisAlignment(osgText::Text::SCREEN);
  textOne->setPosition(osg::Vec3(360, 165, -1.5));
  textOne->setColor(osg::Vec4(199, 77, 15, 1));

  // // Declare a geode to contain the tank's text label:
  // osg::Geode *tankLabelGeode = new osg::Geode();

  // // Add the tank label to the scene:
  // tankLabelGeode->addDrawable(tankLabel);
  // tankXform->addChild(tankLabelGeode);

  // // Set up the parameters for the text label for the tank
  // // align text with tank's SCREEN.
  // // (for Onder: use XZ_PLANE to align text with tank's XZ plane.)
  // tankLabel->setCharacterSize(5);
  // tankLabel->setFont("/fonts/arial.ttf");
  // tankLabel->setText("Tank #1");
  // tankLabel->setAxisAlignment(osgText::Text::XZ_PLANE);

  // // Set the text to render with alignment anchor and bounding box around it:
  // tankLabel->setDrawMode(osgText::Text::TEXT |
  //                        osgText::Text::ALIGNMENT |
  //                        osgText::Text::BOUNDINGBOX);
  // tankLabel->setAlignment(osgText::Text::CENTER_TOP);
  // tankLabel->setPosition(osg::Vec3(0, 0, 8));
  // tankLabel->setColor(osg::Vec4(1.0f, 0.0f, 0.0f, 1.0f));
}

void CFDHud::resize(int width, int height)
{
  HUDProjectionMatrix->setMatrix(osg::Matrix::ortho2D(0, width, 0, height));
}