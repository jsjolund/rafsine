#include <stdio.h>
#include <unistd.h>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <vector>

#include <osg/ArgumentParser>
#include <osg/Vec4>
#include <osgViewer/Viewer>
#include <osgViewer/ViewerEventHandlers>

#include <cuda.h>
#include <cuda_profiler_api.h>

#include <glm/vec3.hpp>

#include "DomainData.hpp"
#include "InputEventHandler.hpp"
#include "VoxelMesh.hpp"

class MyKeyboardHandler : public InputEventHandler {
 private:
  osg::ref_ptr<osg::Group> m_root;
  osg::ref_ptr<VoxelMesh> m_mesh;
  osg::Vec3i m_voxMin, m_voxMax, m_voxSize;

 public:
  explicit MyKeyboardHandler(osg::ref_ptr<osg::Group> root,
                             osg::ref_ptr<VoxelMesh> mesh, osg::Vec3i min,
                             osg::Vec3i max, osg::Vec3i size)
      : m_root(root),
        m_mesh(mesh),
        m_voxMin(min),
        m_voxSize(size),
        m_voxMax(max) {}

  virtual bool keyDown(int key) {
    typedef osgGA::GUIEventAdapter::KeySymbol osgKey;

    switch (key) {
      case osgKey::KEY_F1:
        m_mesh->setPolygonMode(osg::PolygonMode::FILL);
        return true;
      case osgKey::KEY_F2:
        m_mesh->setPolygonMode(osg::PolygonMode::LINE);
        return true;
      case osgKey::KEY_F3:
        m_mesh->setPolygonMode(osg::PolygonMode::POINT);
        return true;
      case osgKey::KEY_F5:
        m_mesh->buildMeshReduced(m_voxMin, m_voxMax);
        return true;
      case osgKey::KEY_F6:
        m_mesh->buildMeshFull(m_voxMin, m_voxMax);
        return true;
      default:
        return false;
    }
  }
};

int main(int argc, char **argv) {
  std::string settings = "problems/data_center/settings.lua";
  std::string geometry = "problems/data_center/geometry.lua";

  LuaData data;
  data.loadFromLua(geometry, settings);
  osg::ref_ptr<VoxelMesh> mesh = new VoxelMesh(data.m_voxGeo->getVoxelArray());
  osg::Vec3i voxSize(mesh->getSizeX(), mesh->getSizeY(), mesh->getSizeZ());
  osg::Vec3i voxMin(-1, -1, -1);
  osg::Vec3i voxMax(voxSize - osg::Vec3i(1, 1, 1));
  mesh->buildMeshFull(voxMin, voxMax);

  osg::ref_ptr<osg::Group> root = new osg::Group;
  root->addChild(mesh);

  osgViewer::Viewer viewer;
  viewer.getCamera()->setClearColor(osg::Vec4(0, 0, 0, 1));
  viewer.setSceneData(root);
  viewer.setUpViewInWindow(400, 400, 800, 600);

  viewer.addEventHandler(
      new MyKeyboardHandler(root, mesh, voxMin, voxMax, voxSize));
  viewer.addEventHandler(new osgViewer::StatsHandler);

  return viewer.run();
}
