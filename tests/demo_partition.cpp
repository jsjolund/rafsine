#include <osg/ArgumentParser>
#include <osg/Vec4>
#include <osgViewer/Viewer>

#include <stdio.h>
#include <unistd.h>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <vector>

#include "DomainData.hpp"
#include "PartitionMesh.hpp"

int main(int argc, char** argv) {
  osg::ArgumentParser args(&argc, argv);

  int divisions = 4;
  int value;
  if (args.read("-d", value)) { divisions = value; }

  std::string settings = "problems/data_center/settings.lua";
  std::string geometry = "problems/data_center/geometry.lua";

  LuaData data;
  data.loadSimulation(geometry, settings);

  osg::ref_ptr<VoxelMesh> voxMesh =
      new VoxelMesh(data.m_voxGeo->getVoxelArray());

  osg::Vec3i voxSize(voxMesh->getSizeX(), voxMesh->getSizeY(),
                     voxMesh->getSizeZ());
  osg::Vec3i voxMin(-1, -1, -1);
  osg::Vec3i voxMax(voxSize - osg::Vec3i(1, 1, 1));

  osg::ref_ptr<PartitionMesh> mesh =
      new PartitionMesh(*voxMesh, divisions, 0.3);

  osg::ref_ptr<osg::Group> root = new osg::Group;
  root->addChild(mesh);

  osgViewer::Viewer viewer;
  viewer.getCamera()->setClearColor(osg::Vec4(0, 0, 0, 1));
  viewer.setSceneData(root);
  viewer.setUpViewInWindow(400, 400, 800, 600);
  return viewer.run();
}
