
#include <osg/ArgumentParser>
#include <osg/Vec4>
#include <osgViewer/Viewer>
#include <osgViewer/ViewerEventHandlers>

#include <cuda.h>
#include <cuda_profiler_api.h>
#include <boost/timer.hpp>
#include <glm/vec3.hpp>

#include <stdio.h>
#include <unistd.h>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <vector>

#include "DomainData.hpp"
#include "InputEventHandler.hpp"
#include "VoxelMesh.hpp"

namespace Axis {
enum Enum { X, Y, Z };
}

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

  void slice(Axis::Enum axis, int inc) {
    if (inc == 0) return;
    int pos;
    switch (axis) {
      case Axis::X:
        pos = m_voxMin.x();
        m_voxMin.x() =
            (pos + inc < m_voxSize.x() && pos + inc >= 0) ? pos + inc : pos;
        break;
      case Axis::Y:
        pos = m_voxMin.y();
        m_voxMin.y() =
            (pos + inc < m_voxSize.y() && pos + inc >= 0) ? pos + inc : pos;
        break;
      case Axis::Z:
        pos = m_voxMax.z();
        m_voxMax.z() =
            (pos + inc < m_voxSize.z() && pos + inc >= 0) ? pos + inc : pos;
        break;
    }
    m_mesh->crop(m_voxMin, m_voxMax);
  }

  virtual bool keyDown(int key) {
    typedef osgGA::GUIEventAdapter::KeySymbol osgKey;
    boost::timer t;
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
        m_mesh->build(VoxelMeshType::REDUCED);
        std::cout << "VoxelMesh build took " << t.elapsed() << " s"
                  << std::endl;
        m_mesh->crop(m_voxMin, m_voxMax);
        return true;
      case osgKey::KEY_F6:
        m_mesh->build(VoxelMeshType::FULL);
        std::cout << "VoxelMesh build took " << t.elapsed() << " s"
                  << std::endl;
        m_mesh->crop(m_voxMin, m_voxMax);
        return true;
      case osgKey::KEY_Page_Down:
        slice(Axis::Z, -1);
        return true;
      case osgKey::KEY_Page_Up:
        slice(Axis::Z, 1);
        return true;
      case osgKey::KEY_End:
        slice(Axis::Y, -1);
        return true;
      case osgKey::KEY_Home:
        slice(Axis::Y, 1);
        return true;
      case osgKey::KEY_Delete:
        slice(Axis::X, -1);
        return true;
      case osgKey::KEY_Insert:
        slice(Axis::X, 1);
        return true;
      default:
        return false;
    }
  }
};

int main(int argc, char **argv) {
  std::string settings = "problems/pod2/settings.lua";
  std::string geometry = "problems/pod2/geometry.lua";

  LuaData data;
  data.loadFromLua(geometry, settings);

  std::cout << "Building voxel mesh..." << std::endl;
  auto start = std::chrono::high_resolution_clock::now();

  osg::ref_ptr<VoxelMesh> mesh = new VoxelMesh(data.m_voxGeo->getVoxelArray());

  auto finish = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> elapsed = finish - start;
  std::cout << "Voxel mesh build time: " << elapsed.count() << " s"
            << std::endl;

  osg::Vec3i voxSize(mesh->getSizeX(), mesh->getSizeY(), mesh->getSizeZ());
  osg::Vec3i voxMin(-1, -1, -1);
  osg::Vec3i voxMax(voxSize - osg::Vec3i(1, 1, 1));

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
