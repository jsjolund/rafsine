#include <stdio.h>
#include <unistd.h>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <vector>

#include <osg/ArgumentParser>
#include <osg/Vec4>
#include <osgViewer/Viewer>

#include <cuda.h>
#include <cuda_profiler_api.h>

#include <glm/vec3.hpp>

#include "DomainData.hpp"
#include "VoxelMesh.hpp"

int main(int argc, char **argv) {
  std::string settings = "problems/data_center/settings.lua";
  std::string geometry = "problems/data_center/geometry.lua";

  LuaData data;
  data.loadFromLua(geometry, settings);
  osg::ref_ptr<VoxelMesh> mesh = new VoxelMesh(data.m_voxGeo->getVoxelArray());
  osg::Vec3i voxSize(mesh->getSizeX(), mesh->getSizeY(), mesh->getSizeZ());
  osg::Vec3i voxMin(-1, -1, -1);
  osg::Vec3i voxMax(voxSize - osg::Vec3i(1, 1, 1));
  mesh->buildMeshReduced(voxMin, voxMax);

  osg::ref_ptr<osg::Group> root = new osg::Group;
  root->addChild(mesh);

  osgViewer::Viewer viewer;
  viewer.getCamera()->setClearColor(osg::Vec4(0, 0, 0, 1));
  viewer.setSceneData(root);
  viewer.setUpViewInWindow(400, 400, 800, 600);
  return viewer.run();
}
