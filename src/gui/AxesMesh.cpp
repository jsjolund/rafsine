#include "AxesMesh.hpp"

void AxesMesh::resize(int width, int height) {
  setPosition(osg::Vec3d(width * 0.1, width * 0.1, -1));
}

AxesMesh::AxesMesh() : osg::PositionAttitudeTransform() {
  osg::ref_ptr<osg::Node> axesModel = osgDB::readNodeFile("assets/axes.osgt");

  osg::ref_ptr<osg::StateSet> stateset = axesModel->getOrCreateStateSet();
  stateset->setMode(GL_LIGHTING,
                    osg::StateAttribute::OVERRIDE | osg::StateAttribute::OFF);
  osg::ref_ptr<osg::Material> mat = new osg::Material;
  mat->setAmbient(osg::Material::Face::FRONT_AND_BACK,
                  osg::Vec4f(1.0f, 1.0f, 1.0f, 1.0f) * 1.0f);
  mat->setDiffuse(osg::Material::Face::FRONT_AND_BACK,
                  osg::Vec4f(1.0f, 1.0f, 1.0f, 1.0f) * 1.0f);
  mat->setColorMode(osg::Material::ColorMode::AMBIENT_AND_DIFFUSE);
  stateset->setAttributeAndModes(
      mat, osg::StateAttribute::OVERRIDE | osg::StateAttribute::ON);

  addChild(axesModel);
  setScale(osg::Vec3d(40, 40, 40));
}
