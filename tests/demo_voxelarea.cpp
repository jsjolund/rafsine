#include <osg/ArgumentParser>
#include <osg/Geode>
#include <osg/Group>
#include <osg/Vec4>
#include <osgViewer/Viewer>

#include <stdio.h>
#include <unistd.h>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <vector>

#include <glm/vec3.hpp>

#include "VoxelAreaMesh.hpp"

int main(int argc, char **argv) {
  glm::ivec3 min(0, 0, 0);
  glm::ivec3 max(5, 10, 2);

  osg::ref_ptr<VoxelAreaMesh> mesh = new VoxelAreaMesh(
      osg::Vec3i(min.x, min.y, min.z), osg::Vec3i(max.x, max.y, max.z));

  osg::ref_ptr<osg::Group> root = new osg::Group;
  osg::ref_ptr<osg::Geode> geode = new osg::Geode();
  geode->addDrawable(mesh);
  root->addChild(geode);

  osgViewer::Viewer viewer;
  viewer.getCamera()->setClearColor(osg::Vec4(0, 0, 0, 1));
  viewer.setSceneData(root);
  viewer.setUpViewInWindow(400, 400, 800, 600);
  return viewer.run();
}
