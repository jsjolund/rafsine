#include "AxesMesh.hpp"

void AxesMesh::resize(int width, int height) {
  setPosition(osg::Vec3d(width * 0.075, width * 0.075, -1));
}

AxesMesh::AxesMesh() : osg::PositionAttitudeTransform() {
  std::vector<std::string> axisModelFiles;
  axisModelFiles.push_back("assets/axis.obj");
  axisModelFiles.push_back("assets/axis.mtl");

  osg::ref_ptr<osg::Node> axesModel = osgDB::readNodeFiles(axisModelFiles);

  osg::ref_ptr<osg::StateSet> stateset = axesModel->getOrCreateStateSet();
  stateset->setMode(GL_LIGHTING,
                    osg::StateAttribute::OVERRIDE | osg::StateAttribute::OFF);

  addChild(axesModel);
  setScale(osg::Vec3d(20, 20, 20));
}
