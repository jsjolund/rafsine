#include "AxesMesh.hpp"

void AxesMesh::resize(int width, int) {
  setPosition(osg::Vec3d(width * 0.075, width * 0.075, -100));
}

AxesMesh::AxesMesh() : osg::PositionAttitudeTransform() {
  std::vector<std::string> axisModelFiles;
  axisModelFiles.push_back("assets/axis.obj");
  axisModelFiles.push_back("assets/axis.mtl");

  osg::ref_ptr<osg::Node> axesModel = osgDB::readNodeFiles(axisModelFiles);

  osg::ref_ptr<osg::StateSet> stateset = axesModel->getOrCreateStateSet();
  stateset->setMode(GL_LIGHTING, osg::StateAttribute::OFF);

  addChild(axesModel);
  setScale(osg::Vec3d(20, 20, 20));

  addChild(createBillboardText(Vector3<int>(2, 0, 1), "x"));
  addChild(createBillboardText(Vector3<int>(0, 2, 1), "y"));
  addChild(createBillboardText(Vector3<int>(0, 0, 3), "z"));
}
